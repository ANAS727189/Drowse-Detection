from __future__ import annotations

import os
from threading import Lock, Thread

try:
    import playsound
except ImportError:
    playsound = None


class AlarmPlayer:
    def __init__(self, alarm_path: str) -> None:
        self.alarm_path = alarm_path
        self._lock = Lock()
        self._alarm_on = False

    def trigger(self) -> None:
        with self._lock:
            if self._alarm_on:
                return
            self._alarm_on = True
        Thread(target=self._play, daemon=True).start()

    def _play(self) -> None:
        try:
            if playsound and os.path.exists(self.alarm_path):
                playsound.playsound(self.alarm_path)
        finally:
            with self._lock:
                self._alarm_on = False

