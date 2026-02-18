# Driver Monitor: Drowsiness + Yawn Detection

It analyzes live video from a webcam, estimates facial landmarks, and raises alerts when signs of drowsiness or fatigue are detected.

## What Makes This Different

- Modular codebase (`driver_monitor/`) instead of one large script
- Dashboard-style UI with live metrics (EAR, mouth opening, yaw, pitch, FPS)
- Multi-condition alerts:
  - drowsiness (prolonged eye closure)
  - yawn detection
  - distraction/head pose warnings
- Command-line controls for webcam source, thresholds, alarm file, and fullscreen mode

## Tech Stack

- [OpenCV](https://opencv.org/) for camera input and rendering
- [TensorFlow / Keras](https://www.tensorflow.org/) for `.h5` model inference
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) for face landmarks
- [NumPy](http://www.numpy.org/) for numeric operations
- [playsound](https://pypi.org/project/playsound/) for alarm playback
- [imutils](https://github.com/jrosebr1/imutils) utilities for OpenCV workflows


## Project Layout

- `drowsiness_yawn.py` - launcher entrypoint
- `driver_monitor/app.py` - app runtime loop and CLI args
- `driver_monitor/analysis.py` - EAR, yawn, and head-pose calculations
- `driver_monitor/ui.py` - dashboard rendering and overlays
- `driver_monitor/audio.py` - alarm playback
- `driver_monitor/config.py` - runtime thresholds and defaults

## Quick Start

1. Clone your repo and move into the project directory.
```bash
git clone <your-repo-url>
cd Real-Time-Drowsiness-Detection-System-main
```

2. Create and activate a virtual environment (recommended).
```bash
python -m venv .venv
source .venv/bin/activate
```
On Windows:
```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies.
```bash
pip install -r requirements.txt
```

4. Run the app.
```bash
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
```

Use a trained `.h5` model:
```bash
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav --model-h5 drowsiness_model.h5 --model-drowsy-class 0 --model-threshold 0.5
```

5. Optional fullscreen start.
```bash
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav --fullscreen
```

## CLI Options

- `--webcam` webcam index (`0`, `1`, ...) or video file path
- `--alarm` path to `.wav` alarm file
- `--ear-threshold` eye-closure threshold
- `--ear-frames` consecutive low-EAR frames before drowsiness alert
- `--yawn-threshold` mouth-opening threshold
- `--model-h5` path to trained Keras `.h5` model
- `--model-threshold` score threshold (0-1) for model drowsy alert
- `--model-drowsy-class` output class index treated as drowsy (`0` for Closed/Open binary models)
- `--model-full-frame` disable eye-crop inference and run image model on full frame
- `--fullscreen` start in fullscreen mode

## `.h5` Model Input Support

The app auto-detects your model input type:

- Rank-2 input (`[batch, features]`): expects exactly 4 features in this order:
  `ear, mouth_open, pitch, yaw`
- Rank-4 input (`[batch, height, width, channels]`): uses full frame resized to model input size
  (`channels` must be `1` or `3`)

For eye-state models trained on cropped eyes (Open/Closed), image mode defaults to MediaPipe eye crops,
then averages left/right-eye drowsy scores.

Class mapping note:
- `image_dataset_from_directory(..., label_mode='binary')` assigns index based on sorted folder names.
- If your folders are `Closed` and `Open`, class index is usually `Closed=0`, `Open=1`.
- In that case keep `--model-drowsy-class 0`.

If no `--model-h5` is passed, the app uses only threshold-based logic (EAR + yawn + head pose).

## Detection Flow

1. Capture frame from camera/video input.
2. Extract face landmarks with MediaPipe.
3. Compute eye aspect ratio (EAR) and mouth opening.
4. Estimate head orientation (yaw/pitch) for distraction checks.
5. Trigger alerts and play alarm when thresholds are crossed.
