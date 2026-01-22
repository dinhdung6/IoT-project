# main.py - Optimized for Raspberry Pi
import os
import io
import json
import time
import threading
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ML
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available - model predictions disabled")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except:
    JOBLIB_AVAILABLE = False
    print("⚠️ Joblib not available")

# Serial communication
try:
    import serial
    SERIAL_AVAILABLE = True
except:
    SERIAL_AVAILABLE = False
    print("⚠️ PySerial not available - sensor input disabled")

# ============= CONFIGURATION =============
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
MODEL_PATH = os.path.join(BASE_DIR, "lstm_parking_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_parking.save")
AREAS_GEO = os.path.join(BASE_DIR, "areas_geo.csv")
PRED_CSV = os.path.join(BASE_DIR, "predicted_free_spots_top3areas.csv")

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

SERIAL_PORT = cfg.get("serial_port", "/dev/ttyACM0")
BAUDRATE = cfg.get("baudrate", 9600)
SENSOR_TO_AREA = cfg.get("sensor_to_area", {})
MAP_CENTER = cfg.get("default_map_center", [-37.8150, 144.9460])
PORT = cfg.get("PORT", 8000)

# ============= FASTAPI SETUP =============
app = FastAPI(title="Smart Parking Predictor")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ============= MODEL LOADING =============
model = None
scaler = None

if TF_AVAILABLE and JOBLIB_AVAILABLE:
    try:
        print("Loading LSTM model...")
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✓ Model and scaler loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        model = None
        scaler = None

# ============= GLOBAL STATE =============
latest_predictions_df = None
sensor_state = {k: False for k in SENSOR_TO_AREA.keys()}
sensor_lock = threading.Lock()

# ============= SERIAL SENSOR READER =============
def parse_sensor_line(line: str):
    """Parse 'Sensor 1 DETECT' or 'Sensor 1 UNDETECT'"""
    try:
        parts = line.strip().split()
        if len(parts) >= 3:
            name = parts[0] + " " + parts[1]
            status = parts[2].upper()
            return name, status
    except:
        pass
    return None, None


def serial_reader_thread():
    if not SERIAL_AVAILABLE:
        return
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        print(f"✓ Serial port open: {SERIAL_PORT}")
    except Exception as e:
        print(f"✗ Serial port failed: {e}")
        return

    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                time.sleep(0.1)
                continue
            
            name, status = parse_sensor_line(line)
            if name and name in SENSOR_TO_AREA:
                with sensor_lock:
                    if status in ["DETECT", "COVER", "COVERED", "OCCUPIED"]:
                        sensor_state[name] = True
                    elif status in ["UNDETECT", "FREE"]:
                        sensor_state[name] = False
                
                print(f"[SENSOR] {name} → {status}")
        except Exception as e:
            print(f"[SERIAL ERROR] {e}")
            time.sleep(1)


# Start serial thread (daemon)
if SERIAL_AVAILABLE:
    threading.Thread(target=serial_reader_thread, daemon=True).start()

# ============= PREDICTION HELPERS =============
def build_ts_free_for_area(df_raw: pd.DataFrame, area_name: str, freq: str = "5min"):
    """Build time series of free spots"""
    df_area = df_raw[df_raw["AreaName"] == area_name].copy()
    total_bays = df_area["BayId"].nunique()
    
    if total_bays == 0:
        return None, 0

    df_area["start"] = pd.to_datetime(df_area["ArrivalTime"]).dt.floor(freq)
    df_area["end"] = pd.to_datetime(df_area["DepartureTime"]).dt.ceil(freq)

    events = pd.concat([
        df_area[["start", "BayId"]].rename(columns={"start": "time"}).assign(change=1),
        df_area[["end", "BayId"]].rename(columns={"end": "time"}).assign(change=-1)
    ], ignore_index=True)
    
    if events.empty:
        return None, total_bays

    ts = events.groupby("time")["change"].sum().sort_index().cumsum()
    ts_free = (total_bays - ts).rename("free_spots")
    ts_free = ts_free.asfreq(freq).interpolate()
    return ts_free, total_bays


def add_time_features(df_ts: pd.DataFrame):
    """Add hour, day_of_week, is_weekend"""
    df = df_ts.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def create_sequences(df_features: pd.DataFrame, window_size: int):
    """Create sliding windows"""
    arr = df_features.values
    X = []
    for i in range(len(arr) - window_size):
        X.append(arr[i:i+window_size])
    return np.stack(X) if X else np.empty((0, window_size, arr.shape[1]))


def predict_for_area(ts_free, window_size=24, horizon=6):
    """Generate predictions using LSTM"""
    if model is None or scaler is None:
        return None, None
    
    df_ts = ts_free.to_frame()
    df_ts = add_time_features(df_ts)
    
    try:
        arr_scaled = scaler.transform(df_ts)
    except Exception as e:
        print(f"Scaler error: {e}")
        return None, None
    
    X = create_sequences(
        pd.DataFrame(arr_scaled, index=df_ts.index, columns=df_ts.columns),
        window_size
    )
    
    if X.shape[0] == 0:
        return None, None
    
    try:
        y_pred_scaled = model.predict(X, verbose=0)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None
    
    # Inverse transform
    temp = np.zeros((len(y_pred_scaled), arr_scaled.shape[1]))
    temp[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(temp)[:, 0]
    
    # Get timestamps
    timestamps = []
    idx = df_ts.index
    for i in range(len(idx) - window_size):
        ts_pred = idx[i + window_size + horizon - 1]
        timestamps.append(ts_pred)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "predicted_free_spots": y_pred
    }), df_ts


# ============= API ENDPOINTS =============
@app.get("/")
async def index():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Process CSV dataset through LSTM model"""
    global latest_predictions_df
    
    if model is None or scaler is None:
        return JSONResponse({
            "error": "Model not loaded. Check server logs."
        }, status_code=500)
    
    content = await file.read()
    
    try:
        df = pd.read_csv(
            io.BytesIO(content),
            usecols=["AreaName", "BayId", "ArrivalTime", "DepartureTime"],
            low_memory=False
        )
    except Exception as e:
        return JSONResponse({
            "error": "CSV must have: AreaName, BayId, ArrivalTime, DepartureTime",
            "details": str(e)
        }, status_code=400)
    
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], errors='coerce')
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], errors='coerce')
    df = df.dropna(subset=['ArrivalTime', 'DepartureTime'])
    
    # Get top 3 areas
    top_areas = df['AreaName'].value_counts().head(3).index.tolist()
    if len(top_areas) < 3:
        return JSONResponse({
            "error": "Dataset must have at least 3 areas"
        }, status_code=400)

    results = []
    for area in top_areas:
        ts_free, total_bays = build_ts_free_for_area(df, area)
        if ts_free is None:
            continue
        
        pred_df, _ = predict_for_area(ts_free)
        if pred_df is None:
            continue
        
        # Round up, floor at 0
        pred_df['predicted_free_spots'] = np.ceil(pred_df['predicted_free_spots']).astype(int)
        pred_df.loc[pred_df['predicted_free_spots'] < 0, 'predicted_free_spots'] = 0
        pred_df['AreaName'] = area
        results.append(pred_df[['AreaName', 'timestamp', 'predicted_free_spots']])

    if not results:
        return JSONResponse({
            "error": "No predictions generated"
        }, status_code=500)

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(PRED_CSV, index=False)
    latest_predictions_df = df_all.copy()
    
    return FileResponse(
        PRED_CSV,
        media_type='text/csv',
        filename='predicted_free_spots_top3areas.csv'
    )


@app.post("/upload_prediction_csv")
async def upload_prediction_csv(file: UploadFile = File(...)):
    """Upload pre-calculated predictions"""
    global latest_predictions_df
    
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse({
            "error": f"Invalid CSV: {str(e)}"
        }, status_code=400)
    
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    
    if not {'areaname', 'timestamp', 'predicted_free_spots'}.issubset(set(df.columns)):
        return JSONResponse({
            "error": "CSV needs: AreaName, timestamp, predicted_free_spots"
        }, status_code=400)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    latest_predictions_df = df.copy()
    latest_predictions_df.to_csv(PRED_CSV, index=False)
    
    return {"message": "Predictions loaded"}


@app.post("/upload_geo")
async def upload_geo(file: UploadFile = File(...)):
    """Upload area coordinates (AreaName, lat, lon)"""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
        
        if not {'AreaName', 'lat', 'lon'}.issubset(set(df.columns)):
            return JSONResponse({
                "error": "CSV needs: AreaName, lat, lon"
            }, status_code=400)
        
        df.to_csv(AREAS_GEO, index=False)
        return {"message": "Geo data saved"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/get_predictions")
async def get_predictions():
    """Get latest predictions with sensor overrides"""
    global latest_predictions_df, sensor_state
    
    if latest_predictions_df is None or latest_predictions_df.empty:
        return JSONResponse({
            "error": "No predictions. Upload dataset or CSV first."
        }, status_code=400)

    df = latest_predictions_df.copy()
    df.columns = [c.lower() for c in df.columns]
    df['predicted_free_spots'] = pd.to_numeric(
        df['predicted_free_spots'], errors='coerce'
    ).fillna(0).astype(int).clip(lower=0)

    # Load geo
    geo = None
    if os.path.exists(AREAS_GEO):
        try:
            geo = pd.read_csv(AREAS_GEO)
        except:
            pass

    out = []
    for idx, area in enumerate(df['areaname'].unique()):
        df_area = df[df['areaname'] == area].sort_values('timestamp', ignore_index=True)
        if df_area.empty:
            continue
        
        last_row = df_area.iloc[-1]
        pred_val = int(last_row['predicted_free_spots'])
        ts_val = last_row['timestamp']

        # Sensor override: if any sensor for this area detected → full
        is_covered = False
        with sensor_lock:
            for sname, mapped_area in SENSOR_TO_AREA.items():
                if mapped_area == area and sensor_state.get(sname, False):
                    is_covered = True
                    break
        
        if is_covered:
            pred_val = 0

        # Get coordinates
        lat, lon = None, None
        if geo is not None:
            try:
                row = geo[geo['AreaName'] == area]
                if not row.empty:
                    lat = float(row.iloc[0]['lat'])
                    lon = float(row.iloc[0]['lon'])
            except:
                pass
        
        # Fallback
        if lat is None or lon is None:
            lat = MAP_CENTER[0] + 0.003 * idx
            lon = MAP_CENTER[1] + 0.003 * idx

        out.append({
            "AreaName": area,
            "timestamp": str(ts_val),
            "predicted_free_spots": pred_val,
            "lat": lat,
            "lon": lon,
        })

    return {"areas": out}


@app.get("/sensor_status")
async def sensor_status():
    """Debug: get sensor states"""
    with sensor_lock:
        return dict(sensor_state)


@app.get("/download_predictions")
async def download_predictions():
    """Download predictions CSV"""
    if not os.path.exists(PRED_CSV):
        return JSONResponse({
            "error": "No predictions available"
        }, status_code=404)
    
    return FileResponse(
        PRED_CSV,
        media_type='text/csv',
        filename='predicted_free_spots_top3areas.csv'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)