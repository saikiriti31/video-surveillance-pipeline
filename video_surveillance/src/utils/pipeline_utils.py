"""
Pipeline utilities: video info extraction, frame sampling, performance stats.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str


def get_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    return VideoInfo(
        path=video_path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        codec=codec,
    )


class FrameSampler:
    """
    Controls which frames are processed to manage compute budget.

    Strategies:
      - 'every_n': process every N frames (default N=1 = all frames)
      - 'target_fps': downsample to target FPS (e.g. 10fps on 30fps video)
    """

    def __init__(self, source_fps: float, target_fps: Optional[float] = None, every_n: int = 1):
        self.source_fps = source_fps
        if target_fps and target_fps < source_fps:
            self.every_n = max(1, round(source_fps / target_fps))
        else:
            self.every_n = max(1, every_n)

        self.effective_fps = source_fps / self.every_n
        logger.info(
            f"FrameSampler: process every {self.every_n} frames "
            f"→ effective {self.effective_fps:.1f} fps"
        )

    def should_process(self, frame_number: int) -> bool:
        return frame_number % self.every_n == 0


@dataclass
class PipelineStats:
    """Collects and reports pipeline performance metrics."""
    video_path: str
    start_time: float = field(default_factory=time.perf_counter)
    frames_processed: int = 0
    frames_skipped: int = 0
    total_detections: int = 0
    inference_times_ms: list = field(default_factory=list)

    def record_frame(self, n_detections: int, inference_ms: float):
        self.frames_processed += 1
        self.total_detections += n_detections
        self.inference_times_ms.append(inference_ms)

    def record_skip(self):
        self.frames_skipped += 1

    def summary(self) -> dict:
        elapsed = time.perf_counter() - self.start_time
        avg_inf = (
            sum(self.inference_times_ms) / len(self.inference_times_ms)
            if self.inference_times_ms else 0.0
        )
        return {
            "video": self.video_path,
            "elapsed_seconds": round(elapsed, 2),
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "pipeline_fps": round(self.frames_processed / elapsed if elapsed > 0 else 0, 2),
            "avg_inference_ms": round(avg_inf, 2),
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(
                self.total_detections / max(self.frames_processed, 1), 2
            ),
        }

    def log_summary(self):
        s = self.summary()
        logger.info("=" * 60)
        logger.info("PIPELINE PERFORMANCE SUMMARY")
        for k, v in s.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)
