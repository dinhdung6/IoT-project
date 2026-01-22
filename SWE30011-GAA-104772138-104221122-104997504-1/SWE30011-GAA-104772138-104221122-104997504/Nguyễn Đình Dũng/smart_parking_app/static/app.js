// ============= CONFIGURATION =============
// Hardcoded user location (Raspberry Pi location - Melbourne)
const USER_LOCATION = { lat: -37.8150, lng: 144.9460 };
const MAP_CENTER = [-37.8150, 144.9460];
const MAP_ZOOM = 15;

// ============= STATE =============
let map = null;
let markers = {}; // AreaName -> marker object
let routingControl = null;
let latestPredictions = []; // Array of {AreaName, timestamp, predicted_free_spots, lat, lon}
let allAreasData = {}; // Cache for area data
let currentPriority = 1; // 1: closest, 2: most available, 3: only available left, 4: manual
let currentRoutedArea = null;

// ============= UI UTILITIES =============
function showStatus(msg, type = 'info') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = `show ${type}`;
  
  if (type !== 'error') {
    setTimeout(() => el.classList.remove('show'), 4000);
  }
}

function getMarkerColor(spots) {
  if (spots <= 0) return '#000000'; // black
  if (spots < 100) return '#dc3545'; // red
  if (spots < 200) return '#ffc107'; // yellow
  return '#28a745'; // green
}

function getStatusText(spots) {
  if (spots <= 0) return '❌ FULL';
  if (spots < 100) return '⚠️ Few';
  if (spots < 200) return '⚡ Some';
  return '✅ Many';
}

function calculateDistance(lat, lng) {
  const from = L.latLng(USER_LOCATION.lat, USER_LOCATION.lng);
  const to = L.latLng(lat, lng);
  return from.distanceTo(to);
}

// ============= MAP INITIALIZATION =============
function initMap() {
  map = L.map('map', { 
    attributionControl: true,
    zoomControl: true
  }).setView(MAP_CENTER, MAP_ZOOM);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 19,
    minZoom: 12
  }).addTo(map);

  // User location marker
  const userMarker = L.circleMarker([USER_LOCATION.lat, USER_LOCATION.lng], {
    radius: 8,
    color: '#0066ff',
    fillColor: '#0066ff',
    fillOpacity: 0.8,
    weight: 2
  }).addTo(map);
  
  userMarker.bindPopup('📍 Your Location (Fixed)');
}

// ============= ROUTING =============
function drawRoute(toLat, toLng, areaName) {
  if (routingControl) {
    map.removeControl(routingControl);
  }
  
  try {
    routingControl = L.Routing.control({
      waypoints: [
        L.latLng(USER_LOCATION.lat, USER_LOCATION.lng),
        L.latLng(toLat, toLng)
      ],
      routeWhileDragging: false,
      show: true,
      collapsible: true,
      lineOptions: { addWaypoints: false },
      router: L.Routing.osrmv1({
        serviceUrl: 'https://router.project-osrm.org/route/v1'
      })
    }).addTo(map);
    
    showStatus(`🗺️ Route to ${areaName}`, 'success');
  } catch (e) {
    showStatus('Routing failed: ' + e.message, 'error');
    console.error(e);
  }
}

window.routeTo = function(areaName) {
  const area = latestPredictions.find(a => a.AreaName === areaName);
  if (!area) {
    showStatus('Area not found', 'error');
    return;
  }
  drawRoute(area.lat, area.lon, areaName);
};

window.routeToMostAvailable = function() {
  if (!latestPredictions || latestPredictions.length === 0) {
    showStatus('No prediction data', 'error');
    return;
  }
  
  // Find area with most available spots
  const best = latestPredictions.reduce((a, b) => 
    (b.predicted_free_spots || 0) > (a.predicted_free_spots || 0) ? b : a
  );
  
  drawRoute(best.lat, best.lon, best.AreaName + ` (${best.predicted_free_spots} spots)`);
};

function updateAutoRoute() {
  if (!latestPredictions || latestPredictions.length === 0) return;

  const available = latestPredictions.filter(a => a.predicted_free_spots > 0);

  if (available.length === 0) {
    showStatus('No available areas, switching to manual', 'error');
    if (routingControl) map.removeControl(routingControl);
    routingControl = null;
    currentRoutedArea = null;
    currentPriority = 4;
    return;
  }

  // Check if current route is still valid
  let needReroute = true;
  if (currentRoutedArea) {
    const current = latestPredictions.find(a => a.AreaName === currentRoutedArea);
    if (current && current.predicted_free_spots > 0) {
      needReroute = false;
    }
  }

  if (!needReroute) return;

  // Advance priority since current became invalid
  currentPriority += 1;
  if (currentPriority > 4) currentPriority = 4;

  let selected = null;
  let label = '';

  if (currentPriority === 1) {
    const sorted = [...available].sort((a, b) => calculateDistance(a.lat, a.lon) - calculateDistance(b.lat, b.lon));
    selected = sorted[0];
    label = ' (Closest available)';
  } else if (currentPriority === 2) {
    selected = available.reduce((a, b) => (b.predicted_free_spots > a.predicted_free_spots) ? b : a);
    label = ' (Most available)';
  } else if (currentPriority === 3) {
    if (available.length === 1) {
      selected = available[0];
      label = ' (Only available)';
    }
  } else {
    // Priority 4: manual - clear route
    if (routingControl) map.removeControl(routingControl);
    routingControl = null;
    currentRoutedArea = null;
    showStatus('Switched to manual routing', 'info');
    return;
  }

  if (selected) {
    drawRoute(selected.lat, selected.lon, selected.AreaName + label);
    currentRoutedArea = selected.AreaName;
    showStatus(`Auto routing to priority ${currentPriority}: ${selected.AreaName + label}`, 'success');
  } else {
    // No selection possible at this priority, advance and retry
    currentPriority += 1;
    if (currentPriority > 4) currentPriority = 4;
    updateAutoRoute();
  }
}

// ============= RENDERING =============
function renderMarkers() {
  if (!map) return;
  
  // Clear old markers
  Object.values(markers).forEach(marker => map.removeLayer(marker));
  markers = {};

  // Add new markers
  latestPredictions.forEach(area => {
    const { AreaName, lat, lon, predicted_free_spots } = area;
    const color = getMarkerColor(predicted_free_spots);
    
    const marker = L.circleMarker([lat, lon], {
      radius: 16,
      color: color,
      fillColor: color,
      fillOpacity: 0.8,
      weight: 2
    }).addTo(map);
    
    const popupText = `
      <div style="font-size:12px;">
        <b>${AreaName}</b><br/>
        Free: <strong>${predicted_free_spots}</strong><br/>
        ${getStatusText(predicted_free_spots)}<br/>
        <button onclick="routeTo('${AreaName}')" style="margin-top:6px; padding:4px 8px; width:100%; background:#667eea; color:white; border:none; border-radius:3px; cursor:pointer;">Route</button>
      </div>
    `;
    
    marker.bindPopup(popupText);
    markers[AreaName] = marker;
  });
}

function renderAreaCards() {
  const container = document.getElementById('areaInfo');
  container.innerHTML = '';

  latestPredictions.forEach(area => {
    const { AreaName, predicted_free_spots } = area;
    const color = getMarkerColor(predicted_free_spots);
    const status = getStatusText(predicted_free_spots);

    const card = document.createElement('div');
    card.className = 'area-card';
    
    card.innerHTML = `
      <h4>
        <span class="status-dot" style="background-color: ${color};"></span>
        ${AreaName}
      </h4>
      <div class="spots">${predicted_free_spots}</div>
      <p style="color: ${color}; font-weight: 600;">${status}</p>
      <button onclick="routeTo('${AreaName}')">📍 Route here</button>
      <button onclick="routeToMostAvailable()" style="margin-top: 4px;">⭐ Best spot</button>
    `;
    
    container.appendChild(card);
  });
}

// ============= DATA FETCHING =============
async function loadPredictions() {
  try {
    const res = await fetch('/get_predictions');
    if (!res.ok) {
      const data = await res.json();
      showStatus('Error: ' + (data.error || 'Failed to fetch'), 'error');
      return;
    }
    
    const data = await res.json();
    latestPredictions = data.areas || [];
    
    if (latestPredictions.length === 0) {
      showStatus('No predictions available', 'error');
      return;
    }
    
    renderMarkers();
    renderAreaCards();
    showStatus(`✅ Loaded ${latestPredictions.length} areas`, 'success');
  } catch (e) {
    showStatus('Failed: ' + e.message, 'error');
    console.error(e);
  }
}

// ============= FILE UPLOADS =============
document.getElementById('uploadBtn').onclick = async () => {
  const file = document.getElementById('datasetFile').files[0];
  if (!file) {
    showStatus('Select a CSV file', 'error');
    return;
  }

  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Processing...';

  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch('/upload_dataset', {
      method: 'POST',
      body: formData
    });
    
    if (!res.ok) {
      const data = await res.json();
      showStatus('Error: ' + (data.error || 'Upload failed'), 'error');
      return;
    }
    
    // Download predictions
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predicted_free_spots_top3areas.csv';
    a.click();
    URL.revokeObjectURL(url);
    
    showStatus('✅ Dataset processed! CSV downloaded.', 'success');
    await loadPredictions();
  } catch (e) {
    showStatus('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run LSTM Model';
  }
};

document.getElementById('uploadPredBtn').onclick = async () => {
  const file = document.getElementById('predFile').files[0];
  if (!file) {
    showStatus('Select a predictions CSV', 'error');
    return;
  }

  const btn = document.getElementById('uploadPredBtn');
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch('/upload_prediction_csv', {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    if (data.error) {
      showStatus('Error: ' + data.error, 'error');
    } else {
      showStatus('✅ Predictions loaded!', 'success');
      await loadPredictions();
    }
  } catch (e) {
    showStatus('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('uploadGeoBtn').onclick = async () => {
  const file = document.getElementById('geoFile').files[0];
  if (!file) {
    showStatus('Select areas_geo.csv', 'error');
    return;
  }

  const btn = document.getElementById('uploadGeoBtn');
  btn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch('/upload_geo', {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    if (data.error) {
      showStatus('Error: ' + data.error, 'error');
    } else {
      showStatus('✅ Geo data updated!', 'success');
      await loadPredictions();
    }
  } catch (e) {
    showStatus('Error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
};

document.getElementById('applyTime').onclick = async () => {
  const minutes = parseInt(document.getElementById('manualTime').value || '30', 10);
  
  if (minutes < 5 || minutes > 60) {
    showStatus('⚠️ Time must be 5-60 minutes', 'error');
    return;
  }
  
  // Backend does interpolation automatically
  // Just refresh to show effects
  await loadPredictions();
};

document.getElementById('downloadBtn').onclick = () => {
  window.location.href = '/download_predictions';
};

document.getElementById('refreshBtn').onclick = loadPredictions;

// ============= INITIALIZATION =============
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadPredictions();
  
  // Auto-refresh every 10 minutes (600000ms)
  setInterval(loadPredictions, 600000);
});

document.getElementById('autoRouteBtn').onclick = () => {
  currentPriority = 1;
  updateAutoRoute();
};