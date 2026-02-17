from __future__ import annotations

from dataclasses import dataclass
from typing import Union

WebcamSource = Union[int, str]


@dataclass(frozen=True)
class MonitorConfig:
    webcam_source: WebcamSource = 0
    alarm_path: str = "Alert.wav"
    ear_threshold: float = 0.30
    ear_frames: int = 15
    yawn_threshold: float = 25.0
    yaw_alert_threshold: float = 20.0
    pitch_down_threshold: float = -15.0
    window_title: str = "Driver Monitor Dashboard"
    start_fullscreen: bool = False

