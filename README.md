# Real-Time-Drowsiness-Detection-System

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving. The objective of this project is to build a drowsiness detection system that will detect drowsiness through the implementation of computer vision system that automatically detects drowsiness in real-time from a live video stream and then alert the user with an alarm notification.

## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) - Real-time face landmark tracking used for eye/yawn geometry.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Install and set up Python 3.

## Running the application

1. Clone the repository. 

    ```
    git clone https://github.com/AnshumanSrivastava108/Real-Time-Drowsiness-Detection-System
    ```
    
1. Move into the project directory. 

    ```
    cd Real-Time-Drowsiness-Detection-System
    ```
 
1. (Optional) Running it in a virtual environment. 

   1. Downloading and installing _virtualenv_. 
   ```
   pip install virtualenv
   ```
   
   2. Create the virtual environment in Python 3.
   
   ```
    virtualenv -p C:\Python37\python.exe test_env
   ```    
   
   3. Activate the test environment.     
   
        1. For Windows:
        ```
        test_env\Scripts\Activate
        ```        
        
        2. For Unix:
        ```
        source test_env/bin/activate
        ```    

1. Install all the required libraries, by installing the requirements.txt file.

    ```
    pip install -r requirements.txt
    ```
    
1. Run the application.

    ```
    python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
    ```

1. Optional flags:

    ```
    python drowsiness_yawn.py --webcam 0 --alarm Alert.wav --fullscreen
    ```

## Code Structure

The project is now split into small modules so logic is not crowded in one file:

- `drowsiness_yawn.py` -> thin launcher/entrypoint
- `driver_monitor/app.py` -> runtime loop + CLI args
- `driver_monitor/analysis.py` -> MediaPipe + EAR/yawn/head-pose calculations
- `driver_monitor/ui.py` -> dashboard renderer and overlay visuals
- `driver_monitor/audio.py` -> alarm playback thread
- `driver_monitor/config.py` -> thresholds and runtime settings

## Alogorithm

1. Capture the image of the driver from the webcam.
2. Run MediaPipe Face Mesh on each frame and extract face landmarks.
3. Compute Eye Aspect Ratio (EAR) from eye landmarks.
4. If EAR is below threshold for consecutive frames, trigger drowsiness alert.
5. Compute lip opening distance from mouth landmarks.
6. If lip distance crosses threshold, trigger yawn alert.

