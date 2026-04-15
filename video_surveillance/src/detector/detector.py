"""
PersonDetector — YOLOv9 + ByteTrack

Design choices:
- YOLOv9 (GELAN + PGI architecture) for superior accuracy on occluded/small people
- Model sizes: yolov9t (tiny/CPU), yolov9s (small), yolov9m (medium/GPU recommended)
- ByteTrack integrated natively in Ultralytics; handles low-confidence buffer
  for re-identification when a person re-enters the frame
- Class filter: only COCO class 0 (person) to reduce false positives
- Half-precision (FP16) inference on CUDA for ~2x throughput
- Frame enhancement pipeline for night/rain/fog/occlusion scenarios
- Test Time Augmentation (augment=True) for partially hidden people

Why YOLOv9 over older YOLO versions:
- PGI (Programmable Gradient Information) prevents information loss through layers
- GELAN architecture extracts richer features at same speed
- Better on partially occluded / small people (important for CCTV)
- Native Ultralytics support — same API, just different model weights
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single person detection with tracking ID."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    confidence: float
    frame_number: int
    timestamp: float                   # seconds from video start

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-center of bounding box — best proxy for ground position."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4),
        }


class PersonDetector:
    """
    Wraps YOLOv9 + ByteTrack for person detection and multi-object tracking.

    Parameters
    ----------
    model_size : str
        'n' | 's' | 'm' | 'l' | 'x'  — trades speed vs accuracy.
        Recommended: 'n' for CPU-only, 'm' for GPU.
    confidence : float
        Detection confidence threshold (0.0–1.0). Default 0.3.
    iou_threshold : float
        NMS IoU threshold. Default 0.45.
    device : str
        'cpu' | 'cuda' | 'cuda:0' | 'mps'.  Auto-detected if None.
    half : bool
        FP16 inference. Only effective on CUDA.
    """

    PERSON_CLASS_ID = 0  # COCO class 0 = person

    # YOLOv9 model size map
    # t = tiny  (fastest, CPU-friendly)
    # s = small (balanced)
    # m = medium (best accuracy/speed for GPU)
    # c = compact (between m and e)
    # e = extended (most accurate, needs strong GPU)
    YOLOV9_SIZES = {
        "t": "yolov9t.pt",
        "s": "yolov9s.pt",
        "m": "yolov9m.pt",
        "c": "yolov9c.pt",
        "e": "yolov9e.pt",
    }

    def __init__(
        self,
        model_size: str = "t",
        confidence: float = 0.2,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half: bool = False,
        enhance: bool = False,
        augment: bool = False,
    ):
        self.model_size = model_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.half = half
        self.enhance = enhance
        self.augment = augment

        self._device = self._resolve_device(device)
        self._model = self._load_model()

        logger.info(
            f"PersonDetector ready | model={self.YOLOV9_SIZES.get(model_size, 'yolov9t.pt')} "
            f"device={self._device} conf={confidence} "
            f"enhance={enhance} augment={augment} half={half}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_number: int, fps: float) -> List[Detection]:
        """
        Run detection + tracking on a single BGR frame.

        Returns a list of Detection objects (one per tracked person).
        Returns empty list on failure (defensive — never crash the pipeline).
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Empty frame at #{frame_number} — skipping")
            return []

        # Apply enhancement if enabled (night/rain/fog/occlusion scenes)
        input_frame = self._enhance_frame(frame) if self.enhance else frame

        try:
            results = self._model.track(
                source=input_frame,
                persist=True,            # keeps tracker state across calls
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                iou=self.iou_threshold,
                tracker="bytetrack.yaml",
                augment=self.augment,    # TTA — helps with partial occlusion
                half=self.half,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"Inference failed at frame #{frame_number}: {exc}")
            return []

        return self._parse_results(results, frame_number, fps)

    def reset_tracker(self):
        """Call when switching to a new video to clear ByteTrack state."""
        # Re-loading model resets all tracker state cleanly
        self._model = self._load_model()
        logger.info("Tracker state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            from ultralytics import YOLO

            # Resolve YOLOv9 model filename
            model_name = self.YOLOV9_SIZES.get(self.model_size, "yolov9t.pt")

            logger.info(f"Loading {model_name} ...")
            model = YOLO(model_name)
            # Weights auto-downloaded from Ultralytics hub on first run

            # Warm-up: pre-compile on device (reduces first-frame latency)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(dummy, verbose=False, device=self._device)
            logger.info(f"Loaded {model_name} and warmed up on {self._device}")
            return model
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Multi-stage frame enhancement pipeline for difficult CCTV conditions.

        Stages (applied in order):
          1. Brightness + contrast boost  — helps dark/night scenes
          2. CLAHE                        — improves local contrast in shadows
          3. Bilateral filter             — denoise while keeping edges sharp

        Note: fastNlMeansDenoisingColored is too slow per-frame (~200ms).
              Bilateral filter is used instead (~3ms) with comparable quality.
        """
        try:
            # Stage 1 — Brightness/contrast boost
            # alpha=1.3 → 30% contrast increase, beta=20 → slight brightness lift
            enhanced = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

            # Stage 2 — CLAHE on L channel (LAB color space)
            # Improves contrast in locally dark regions (shadows, night)
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Stage 3 — Bilateral filter (edge-preserving denoise)
            # d=9: pixel neighborhood, sigmaColor/sigmaSpace=75: filter strength
            enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

            return enhanced

        except Exception as exc:
            logger.warning(f"Frame enhancement failed: {exc} — using original")
            return frame  # always fall back to original frame safely

    def _resolve_device(self, device: Optional[str]) -> str:
        if device:
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _parse_results(
        self, results, frame_number: int, fps: float
    ) -> List[Detection]:
        detections: List[Detection] = []
        timestamp = frame_number / fps if fps > 0 else 0.0

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes

            # ByteTrack assigns track IDs; may be None if tracking failed
            if boxes.id is None:
                continue

            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

            for track_id, box, conf in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = box
                detections.append(
                    Detection(
                        track_id=int(track_id),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        frame_number=frame_number,
                        timestamp=timestamp,
                    )
                )

        return detections
