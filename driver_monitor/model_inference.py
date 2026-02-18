from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .analysis import AnalysisResult


@dataclass
class ModelPrediction:
    is_drowsy: bool
    score: float


class H5DrowsinessModel:
    """Keras .h5 inference helper with support for feature and image models."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        drowsy_class: int = 0,
        use_eye_crops: bool = True,
    ) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorFlow is required for .h5 inference. Install it with: pip install tensorflow"
            ) from exc

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.threshold = float(np.clip(threshold, 0.0, 1.0))
        self.drowsy_class = int(drowsy_class)
        self.use_eye_crops = bool(use_eye_crops)
        self.has_builtin_rescaling = any(
            layer.__class__.__name__.lower() == "rescaling" for layer in self.model.layers
        )

        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_shape = tuple(input_shape)
        self.mode = self._infer_mode(self.input_shape)

        self.expected_feature_count = 0
        self.image_height = 0
        self.image_width = 0
        self.image_channels = 0

        if self.mode == "features":
            if self.input_shape[1] is None:
                raise ValueError(f"Invalid feature input shape: {self.input_shape}")
            self.expected_feature_count = int(self.input_shape[1])
        else:
            h, w, c = self.input_shape[1], self.input_shape[2], self.input_shape[3]
            if h is None or w is None or c is None:
                raise ValueError(f"Image input shape must be fixed, got: {self.input_shape}")
            self.image_height = int(h)
            self.image_width = int(w)
            self.image_channels = int(c)

    @staticmethod
    def _infer_mode(input_shape: tuple[Any, ...]) -> str:
        if len(input_shape) == 2:
            return "features"
        if len(input_shape) == 4:
            return "image"
        raise ValueError(
            f"Unsupported model input shape {input_shape}. Expected rank-2 or rank-4 input."
        )

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _drowsy_score_from_sample(self, sample_output: Any) -> float:
        sample = np.squeeze(np.asarray(sample_output))
        if np.ndim(sample) == 0:
            positive_prob = float(sample)
            if positive_prob < 0.0 or positive_prob > 1.0:
                positive_prob = self._sigmoid(positive_prob)
            if self.drowsy_class == 1:
                return float(np.clip(positive_prob, 0.0, 1.0))
            if self.drowsy_class == 0:
                return float(np.clip(1.0 - positive_prob, 0.0, 1.0))
            raise ValueError("For binary scalar output, --model-drowsy-class must be 0 or 1.")

        probs = np.ravel(sample).astype(np.float64)
        if probs.size == 0:
            raise ValueError("Model returned empty prediction.")

        if np.any(probs < 0.0) or np.any(probs > 1.0):
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.maximum(np.sum(exp), 1e-8)
        else:
            probs = np.clip(probs, 0.0, 1.0)
            total = float(np.sum(probs))
            if total > 1.0001:
                probs = probs / total

        if self.drowsy_class < 0 or self.drowsy_class >= probs.size:
            raise ValueError(
                f"Drowsy class index {self.drowsy_class} is out of range for output size {probs.size}."
            )
        return float(np.clip(probs[self.drowsy_class], 0.0, 1.0))

    def _build_features(self, result: Optional[AnalysisResult]) -> Optional[np.ndarray]:
        if result is None:
            return None

        features = np.array(
            [result.ear, result.mouth_open, result.pitch, result.yaw],
            dtype=np.float32,
        )
        if self.expected_feature_count != features.shape[0]:
            raise ValueError(
                f"Model expects {self.expected_feature_count} features but app provides "
                f"{features.shape[0]} (ear, mouth_open, pitch, yaw)."
            )
        return features.reshape(1, -1)

    def _preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (self.image_width, self.image_height))
        if self.image_channels == 1:
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        elif self.image_channels == 3:
            processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                f"Unsupported channel count {self.image_channels}. Expected 1 or 3 channels."
            )

        processed = processed.astype(np.float32)
        if not self.has_builtin_rescaling:
            processed = processed / 255.0
        return processed

    def _extract_eye_crops(self, frame: np.ndarray, result: Optional[AnalysisResult]) -> list[np.ndarray]:
        if result is None:
            return []
        if not self.use_eye_crops:
            return []

        crops: list[np.ndarray] = []
        for box in (result.left_eye_box, result.right_eye_box):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
        return crops

    def _predict_scores(self, batch_input: np.ndarray) -> list[float]:
        raw = self.model.predict(batch_input, verbose=0)
        if isinstance(raw, list):
            raw = raw[0]
        raw_array = np.asarray(raw)

        batch_size = int(batch_input.shape[0])
        if batch_size == 1:
            return [self._drowsy_score_from_sample(raw_array)]

        if raw_array.shape[0] != batch_size:
            raise ValueError(
                "Unexpected model output batch shape "
                f"{raw_array.shape} for input batch size {batch_size}."
            )
        return [self._drowsy_score_from_sample(raw_array[i]) for i in range(batch_size)]

    def predict(self, frame: np.ndarray, result: Optional[AnalysisResult]) -> Optional[ModelPrediction]:
        if self.mode == "features":
            model_input = self._build_features(result)
            if model_input is None:
                return None
            scores = self._predict_scores(model_input)
            score = float(scores[0])
            return ModelPrediction(is_drowsy=score >= self.threshold, score=score)

        eye_crops = self._extract_eye_crops(frame, result)
        images = eye_crops if eye_crops else [frame]
        batch = np.stack([self._preprocess_image(img) for img in images], axis=0)
        scores = self._predict_scores(batch)
        score = float(np.mean(scores))
        return ModelPrediction(is_drowsy=score >= self.threshold, score=score)
