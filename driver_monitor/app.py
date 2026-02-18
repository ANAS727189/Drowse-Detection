from __future__ import annotations

import argparse
import time

import cv2

from .analysis import AnalysisResult, FaceAnalyzer
from .audio import AlarmPlayer
from .config import MonitorConfig, WebcamSource
from .model_inference import H5DrowsinessModel, ModelPrediction
from .ui import DashboardRenderer, UISnapshot


class DriverMonitorApp:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.capture = cv2.VideoCapture(config.webcam_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam/video source: {config.webcam_source}")

        self.analyzer = FaceAnalyzer()
        self.alarm = AlarmPlayer(config.alarm_path)
        self.renderer = DashboardRenderer(config)
        self.model: H5DrowsinessModel | None = None
        self.last_model_prediction: ModelPrediction | None = None
        if config.model_h5_path:
            self.model = H5DrowsinessModel(
                config.model_h5_path,
                config.model_threshold,
                config.model_drowsy_class,
                config.model_eye_crops,
            )
            print(
                "-> Loaded model:",
                config.model_h5_path,
                (
                    f"(mode={self.model.mode}, input={self.model.input_shape}, "
                    f"drowsy_class={self.model.drowsy_class}, eye_crops={self.model.use_eye_crops}, "
                    f"threshold={self.model.threshold:.2f})"
                ),
            )

        self.blink_counter = 0
        self.fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.start_time = self.last_frame_time
        self.fullscreen = False

    def _compute_head_state(self, result: AnalysisResult) -> tuple[str, bool]:
        if result.yaw > self.config.yaw_alert_threshold:
            return "Looking Left", True
        if result.yaw < -self.config.yaw_alert_threshold:
            return "Looking Right", True
        if result.pitch < self.config.pitch_down_threshold:
            return "Looking Down", True
        return "Forward", False

    def _snapshot_from_result(self, result: AnalysisResult | None) -> UISnapshot:
        if result is None:
            self.blink_counter = 0
            return UISnapshot(
                face_detected=False,
                status_text="NO FACE",
                status_color=(0, 200, 255),
                head_state="Unknown",
                ear=0.0,
                mouth_open=0.0,
                pitch=0.0,
                yaw=0.0,
                blink_counter=0,
                drowsy=False,
                yawning=False,
                distracted=False,
                nose_point=None,
            )

        if result.ear < self.config.ear_threshold:
            self.blink_counter += 1
        else:
            self.blink_counter = 0

        drowsy = self.blink_counter >= self.config.ear_frames
        yawning = result.mouth_open > self.config.yawn_threshold
        head_state, distracted = self._compute_head_state(result)

        status_text = "ACTIVE"
        status_color = (60, 200, 60)
        if distracted:
            status_text = "DISTRACTION ALERT"
            status_color = (255, 190, 0)
        if yawning:
            status_text = "YAWN DETECTED"
            status_color = (0, 200, 255)
        if drowsy:
            status_text = "DROWSINESS ALERT"
            status_color = (0, 80, 255)

        return UISnapshot(
            face_detected=True,
            status_text=status_text,
            status_color=status_color,
            head_state=head_state,
            ear=result.ear,
            mouth_open=result.mouth_open,
            pitch=result.pitch,
            yaw=result.yaw,
            blink_counter=self.blink_counter,
            drowsy=drowsy,
            yawning=yawning,
            distracted=distracted,
            nose_point=result.nose_point,
        )

    def _update_fps(self) -> None:
        now = time.perf_counter()
        delta = max(now - self.last_frame_time, 1e-6)
        instant_fps = 1.0 / delta
        if self.fps == 0.0:
            self.fps = instant_fps
        else:
            self.fps = (self.fps * 0.9) + (instant_fps * 0.1)
        self.last_frame_time = now

    def _apply_model_prediction(
        self,
        snapshot: UISnapshot,
        prediction: ModelPrediction | None,
    ) -> None:
        self.last_model_prediction = prediction
        if prediction is None:
            return

        if prediction.is_drowsy:
            snapshot.drowsy = True
            snapshot.status_text = f"MODEL DROWSY ({prediction.score:.2f})"
            snapshot.status_color = (0, 80, 255)
        elif snapshot.status_text == "ACTIVE":
            snapshot.status_text = f"MODEL OK ({prediction.score:.2f})"
            snapshot.status_color = (60, 200, 60)

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        try:
            cv2.setWindowProperty(self.config.window_title, cv2.WND_PROP_FULLSCREEN, mode)
        except cv2.error:
            self.fullscreen = False

    def run(self) -> None:
        print("-> System Started. Press 'q' to exit, 'f' for fullscreen, or close the window.")
        cv2.namedWindow(self.config.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_title, 1280, 720)
        if self.config.start_fullscreen:
            self._toggle_fullscreen()

        try:
            while self.capture.isOpened():
                try:
                    visible = cv2.getWindowProperty(self.config.window_title, cv2.WND_PROP_VISIBLE)
                    if visible < 1:
                        break
                except cv2.error:
                    break

                ok, frame = self.capture.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                result = self.analyzer.analyze(frame)
                snapshot = self._snapshot_from_result(result)
                if self.model is not None:
                    try:
                        prediction = self.model.predict(frame, result)
                        self._apply_model_prediction(snapshot, prediction)
                    except Exception as exc:
                        print(f"-> Model inference disabled after error: {exc}")
                        self.model = None
                        self.last_model_prediction = None
                if snapshot.drowsy:
                    self.alarm.trigger()

                self._update_fps()
                uptime = time.perf_counter() - self.start_time
                canvas = self.renderer.render(frame, snapshot, self.fps, uptime)
                cv2.imshow(self.config.window_title, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    self._toggle_fullscreen()
        finally:
            self.capture.release()
            self.analyzer.close()
            cv2.destroyAllWindows()


def _parse_webcam_source(raw: str) -> WebcamSource:
    try:
        return int(raw)
    except ValueError:
        return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time drowsiness and distraction monitor")
    parser.add_argument("--webcam", default="0", help="Webcam index (0,1,2...) or video file path")
    parser.add_argument("--alarm", default="Alert.wav", help="Path to WAV alarm sound")
    parser.add_argument("--ear-threshold", type=float, default=0.30, help="EAR threshold for eye-closure")
    parser.add_argument(
        "--ear-frames",
        type=int,
        default=15,
        help="Consecutive low-EAR frames before drowsiness alert",
    )
    parser.add_argument("--yawn-threshold", type=float, default=25.0, help="Mouth opening threshold")
    parser.add_argument("--model-h5", default=None, help="Path to trained Keras .h5 model")
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.5,
        help="Probability threshold above which model output is treated as drowsy",
    )
    parser.add_argument(
        "--model-drowsy-class",
        type=int,
        default=0,
        help="Model output class index to treat as drowsy (for binary Closed/Open models use 0)",
    )
    parser.add_argument(
        "--model-full-frame",
        action="store_true",
        help="Use full frame for image model inference instead of landmark-based eye crops",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Start window in fullscreen mode")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MonitorConfig(
        webcam_source=_parse_webcam_source(args.webcam),
        alarm_path=args.alarm,
        ear_threshold=args.ear_threshold,
        ear_frames=args.ear_frames,
        yawn_threshold=args.yawn_threshold,
        model_h5_path=args.model_h5,
        model_threshold=args.model_threshold,
        model_drowsy_class=args.model_drowsy_class,
        model_eye_crops=not args.model_full_frame,
        start_fullscreen=args.fullscreen,
    )
    app = DriverMonitorApp(config)
    app.run()
