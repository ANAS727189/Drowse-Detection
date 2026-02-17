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
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) for face landmarks
- [NumPy](http://www.numpy.org/) for numeric operations
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
- `--fullscreen` start in fullscreen mode

## Detection Flow

1. Capture frame from camera/video input.
2. Extract face landmarks with MediaPipe.
3. Compute eye aspect ratio (EAR) and mouth opening.
4. Estimate head orientation (yaw/pitch) for distraction checks.
5. Trigger alerts and play alarm when thresholds are crossed.
