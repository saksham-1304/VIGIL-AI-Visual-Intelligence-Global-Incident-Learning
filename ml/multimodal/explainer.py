from __future__ import annotations

import logging

import numpy as np

from ml.multimodal.prompt_templates import build_prompt

logger = logging.getLogger(__name__)


class MultimodalExplainer:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.model_name = model_name
        self.captioner = None
        try:
            from PIL import Image
            from transformers import pipeline

            self._image_cls = Image
            self.captioner = pipeline("image-to-text", model=model_name)
            logger.info("Loaded BLIP caption model: %s", model_name)
        except Exception as exc:
            logger.warning("BLIP model unavailable, using template-only descriptions: %s", exc)
            self._image_cls = None

    def describe_scene(self, frame: np.ndarray, detections: list[dict], actions: list[str], anomaly_score: float) -> str:
        caption = self._caption(frame)
        labels = [item["label"] for item in detections]
        prompt = build_prompt(caption, labels, actions, anomaly_score)
        _ = prompt

        if anomaly_score >= 0.8:
            urgency = "critical"
        elif anomaly_score >= 0.6:
            urgency = "high"
        elif anomaly_score >= 0.4:
            urgency = "medium"
        else:
            urgency = "low"

        action_text = ", ".join(actions) if actions else "no abnormal actions"
        detection_text = ", ".join(labels[:4]) if labels else "no dominant objects"

        return (
            f"{caption}. Observed: {detection_text}. "
            f"Behavioral cues: {action_text}. "
            f"Estimated urgency is {urgency} (score={anomaly_score:.2f})."
        )

    def _caption(self, frame: np.ndarray) -> str:
        if self.captioner is None or self._image_cls is None:
            return "Live scene monitored for safety events"

        try:
            rgb = frame[:, :, ::-1]
            image = self._image_cls.fromarray(rgb)
            result = self.captioner(image, max_new_tokens=24)
            if result and isinstance(result, list):
                return result[0].get("generated_text", "Scene analysis unavailable")
        except Exception as exc:
            logger.debug("Caption generation failed: %s", exc)

        return "Live scene monitored for safety events"
