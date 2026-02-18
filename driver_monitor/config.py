from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

WebcamSource = Union[int, str]


@dataclass(frozen=True)
class MonitorConfig:
    webcam_source: WebcamSource = 0
    alarm_path: str = "Alert.wav"
    ear_threshold: float = 0.30
    ear_frames: int = 15
    yawn_threshold: float = 25.0
    model_h5_path: Optional[str] = None
    model_threshold: float = 0.5
    model_drowsy_class: int = 0
    model_eye_crops: bool = True
    yaw_alert_threshold: float = 20.0
    pitch_down_threshold: float = -15.0
    window_title: str = "Driver Monitor Dashboard"
    start_fullscreen: bool = False
