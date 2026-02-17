from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import MonitorConfig


Color = tuple[int, int, int]


@dataclass
class UISnapshot:
    face_detected: bool
    status_text: str
    status_color: Color
    head_state: str
    ear: float
    mouth_open: float
    pitch: float
    yaw: float
    blink_counter: int
    drowsy: bool
    yawning: bool
    distracted: bool
    nose_point: Optional[tuple[int, int]]


class DashboardRenderer:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.panel_width = 350

    @staticmethod
    def _blend_rect(
        frame: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: Color,
        alpha: float,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def _draw_status_chip(self, frame: np.ndarray, x: int, y: int, text: str, color: Color) -> None:
        (text_w, text_h), _ = cv2.getTextSize(text, self.font, 0.62, 1)
        width = text_w + 28
        height = text_h + 16
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 14, y + height - 10),
            self.font,
            0.62,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _metric_color(ratio: float) -> Color:
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if ratio > 0.7:
            return (60, 200, 60)
        if ratio > 0.4:
            return (0, 200, 255)
        return (0, 80, 255)

    def _draw_metric_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        value_text: str,
        lower_is_bad: bool = False,
    ) -> None:
        bar_h = 14
        cv2.putText(frame, label, (x, y), self.font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            value_text,
            (x + width - 64, y),
            self.font,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y_bar = y + 10
        cv2.rectangle(frame, (x, y_bar), (x + width, y_bar + bar_h), (55, 55, 55), -1)
        if max_value <= min_value:
            return
        ratio = (value - min_value) / (max_value - min_value)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        quality = 1.0 - ratio if lower_is_bad else ratio
        color = self._metric_color(quality)
        fill_w = int(width * ratio)
        cv2.rectangle(frame, (x, y_bar), (x + fill_w, y_bar + bar_h), color, -1)
        cv2.rectangle(frame, (x, y_bar), (x + width, y_bar + bar_h), (110, 110, 110), 1)

    @staticmethod
    def _attention_score(yaw: float, pitch: float, face_detected: bool) -> float:
        if not face_detected:
            return 0.0
        yaw_load = min(abs(yaw) / 40.0, 1.0) * 65.0
        pitch_load = min(max(-pitch, 0.0) / 25.0, 1.0) * 35.0
        return max(0.0, 100.0 - (yaw_load + pitch_load))

    def _draw_head_compass(self, frame: np.ndarray, center: tuple[int, int], snapshot: UISnapshot) -> None:
        radius = 48
        cv2.circle(frame, center, radius, (110, 110, 110), 1)
        cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (80, 80, 80), 1)
        cv2.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius), (80, 80, 80), 1)
        if not snapshot.face_detected:
            return
        dx = int(np.clip(snapshot.yaw / 35.0, -1.0, 1.0) * (radius - 8))
        dy = int(np.clip(-snapshot.pitch / 25.0, -1.0, 1.0) * (radius - 8))
        point = (center[0] + dx, center[1] + dy)
        cv2.circle(frame, point, 7, snapshot.status_color, -1)
        cv2.circle(frame, point, 12, snapshot.status_color, 1)

    def render(self, frame: np.ndarray, snapshot: UISnapshot, fps: float, uptime_seconds: float) -> np.ndarray:
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        panel_w = min(self.panel_width, max(290, width // 3))
        panel_x = width - panel_w

        self._blend_rect(canvas, (panel_x, 0), (width, height), (18, 24, 34), 0.55)
        self._blend_rect(canvas, (0, height - 42), (width, height), (20, 20, 20), 0.45)

        cv2.putText(
            canvas,
            "Driver Monitor",
            (18, 32),
            self.font,
            0.9,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        self._draw_status_chip(canvas, 18, 44, snapshot.status_text, snapshot.status_color)

        content_x = panel_x + 20
        y = 38
        cv2.putText(canvas, "Live Metrics", (content_x, y), self.font, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        y += 28

        self._draw_metric_bar(
            canvas,
            content_x,
            y,
            panel_w - 40,
            "Eye Aspect Ratio",
            snapshot.ear,
            0.15,
            0.45,
            f"{snapshot.ear:.2f}",
            lower_is_bad=True,
        )
        y += 42
        self._draw_metric_bar(
            canvas,
            content_x,
            y,
            panel_w - 40,
            "Mouth Opening",
            snapshot.mouth_open,
            5.0,
            35.0,
            f"{snapshot.mouth_open:.1f}",
        )
        y += 42
        attention = self._attention_score(snapshot.yaw, snapshot.pitch, snapshot.face_detected)
        self._draw_metric_bar(
            canvas,
            content_x,
            y,
            panel_w - 40,
            "Attention Score",
            attention,
            0.0,
            100.0,
            f"{attention:.0f}%",
        )
        y += 52

        cv2.putText(
            canvas,
            f"Head: {snapshot.head_state}",
            (content_x, y),
            self.font,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 24
        cv2.putText(
            canvas,
            f"Yaw {snapshot.yaw:+.1f} | Pitch {snapshot.pitch:+.1f}",
            (content_x, y),
            self.font,
            0.47,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        y += 24
        cv2.putText(
            canvas,
            f"Blink Frames: {snapshot.blink_counter}",
            (content_x, y),
            self.font,
            0.47,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

        compass_center = (panel_x + panel_w // 2, height - 116)
        self._draw_head_compass(canvas, compass_center, snapshot)
        cv2.putText(
            canvas,
            "Head Map",
            (compass_center[0] - 42, compass_center[1] - 62),
            self.font,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        runtime = int(uptime_seconds)
        mins, secs = divmod(runtime, 60)
        hours, mins = divmod(mins, 60)
        footer = f"FPS {fps:5.1f}   Runtime {hours:02d}:{mins:02d}:{secs:02d}   [Q] Quit  [F] Fullscreen"
        cv2.putText(canvas, footer, (12, height - 14), self.font, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

        return canvas
