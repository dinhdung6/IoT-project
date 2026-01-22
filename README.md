# 🇻🇳 Vietnamese License Plate Recognition - CS117.O21.KHTN


## Introduction

This project aims to build a web application for recognizing Vietnamese license plates, using efficient deep-learning algorithms for accurate and rapid recognition. This system can work on 2 types of license plate in Vietnam, 1 line plates and 2 lines plates.



## Features

- Detect license plates in images
- Recognize characters on Vietnamese license plates
- User-friendly web interface
- High accuracy and performance



## Installation


    
- Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
## Inference

### Web application
- Run the web server:
```bash
python main.py
```
- Open your web browser and go to `localhost:4000`.


- Click `Choose File` button, select the image of the license plate you want to recognize
- Click the `Upload and Recognize` button to process the recognition.

### On image
```bash
python lp_image.py -i test_image/3.jpg
```

### On webcam
```bash
python webcam.py
```

## Training

**Training code for Yolov5:**

Use code in `training folder`
```bash
  training/Plate_detection.ipynb     #for LP_Detection
  training/Letter_detection.ipynb    #for Letter_detection
```
