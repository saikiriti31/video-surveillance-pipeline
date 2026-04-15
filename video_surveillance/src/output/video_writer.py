"""
AnnotatedVideoWriter — draws bounding boxes, track IDs, zone overlays,
and event banners onto frames and writes output video.

Color scheme:
  - Zones: semi-transparent filled polygon in zone color
  - Normal tracks: green bounding box
  - Track in zone: orange bounding box
  - Loitering track: red bounding box + duration label
  - Event banner: top-left text area for recent events
"""

from __future__ import annotations
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..detector.detector import Detection
from ..events.event_engine import Event, EventType
from ..zones.zone_manager import Zone, ZoneManager

logger = logging.getLogger(__name__)

# Color palette (BGR)
COLOR_NORMAL = (0, 255, 0)          # green
COLOR_IN_ZONE = (0, 165, 255)       # orange
COLOR_LOITERING = (0, 0, 255)       # red
COLOR_TEXT = (255, 255, 255)
COLOR_BANNER_BG = (30, 30, 30)
ZONE_ALPHA = 0.25                   # polygon fill transparency


class AnnotatedVideoWriter:
    """
    Writes an annotated output video with:
    - Zone polygon overlays (semi-transparent)
    - Bounding boxes color-coded by event state
    - Track ID labels
    - Event banners (rolling last-N events)
    - Frame number + timestamp HUD
    - FPS counter
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        zone_manager: ZoneManager,
        banner_history: int = 5,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.zone_manager = zone_manager
        self.banner_history = banner_history

        # Rolling event banner (deque for O(1) append)
        self._event_banner: deque = deque(maxlen=banner_history)
        # Track → active event type for color coding
        self._track_event_state: Dict[int, EventType] = {}

        self._writer = self._init_writer()
        self._frame_times: deque = deque(maxlen=30)
        logger.info(f"VideoWriter ready: {output_path} @ {fps:.1f}fps {width}x{height}")

    def _init_writer(self) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (self.width, self.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {self.output_path}")
        return writer

    def update_events(self, events: List[Event]):
        """Call before write_frame to register events for banner + color coding."""
        for event in events:
            if event.event_type != EventType.ZONE_EXIT:
                self._track_event_state[event.track_id] = event.event_type
                banner_text = (
                    f"[{event.timestamp:.1f}s] "
                    f"{event.event_type.value.upper()} | "
                    f"Zone={event.zone_label} | Track#{event.track_id}"
                )
                if event.duration_seconds:
                    banner_text += f" | {event.duration_seconds:.0f}s"
                self._event_banner.appendleft(banner_text)
            else:
                # Remove event state when track exits zone
                self._track_event_state.pop(event.track_id, None)

    def write_frame(self, frame: np.ndarray, detections: List[Detection], frame_number: int):
        """Annotate and write a single frame."""
        import time
        t0 = time.perf_counter()

        annotated = frame.copy()

        # 1. Draw zone overlays
        self._draw_zones(annotated)

        # 2. Draw detections
        for det in detections:
            self._draw_detection(annotated, det)

        # 3. Draw event banner
        self._draw_banner(annotated)

        # 4. Draw HUD (frame info + FPS)
        self._draw_hud(annotated, frame_number)

        self._writer.write(annotated)

        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)

    def close(self):
        self._writer.release()
        logger.info(f"Video saved: {self.output_path}")
        if self._frame_times:
            avg_ms = (sum(self._frame_times) / len(self._frame_times)) * 1000
            logger.info(f"Avg annotation time per frame: {avg_ms:.1f}ms")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_zones(self, frame: np.ndarray):
        overlay = frame.copy()
        for zone in self.zone_manager.zones:
            pts = zone.numpy_polygon()
            color = zone.color  # BGR

            # Semi-transparent fill
            cv2.fillPoly(overlay, [pts], color)

            # Solid border
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            # Zone label at centroid
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(
                frame, zone.label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        # Blend overlay for transparency
        cv2.addWeighted(overlay, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, frame)

    def _draw_detection(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.bbox
        event_state = self._track_event_state.get(det.track_id)

        if event_state == EventType.LOITERING:
            color = COLOR_LOITERING
        elif event_state == EventType.ZONE_INTRUSION:
            color = COLOR_IN_ZONE
        else:
            color = COLOR_NORMAL

        # Bounding box
        thickness = 3 if event_state else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: "ID:N | 0.92"
        label = f"ID:{det.track_id} | {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Loitering badge
        if event_state == EventType.LOITERING:
            cv2.putText(
                frame, "LOITERING",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LOITERING, 2, cv2.LINE_AA
            )

        # Foot-point dot
        fx, fy = det.foot_point
        cv2.circle(frame, (int(fx), int(fy)), 4, color, -1)

    def _draw_banner(self, frame: np.ndarray):
        if not self._event_banner:
            return
        x, y = 10, 30
        for i, text in enumerate(self._event_banner):
            row_y = y + i * 22
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x - 2, row_y - th - 3), (x + tw + 4, row_y + 3), COLOR_BANNER_BG, -1)
            cv2.putText(frame, text, (x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, frame_number: int):
        h, w = frame.shape[:2]
        fps_display = (
            1.0 / (sum(self._frame_times) / max(len(self._frame_times), 1))
            if self._frame_times else 0.0
        )
        ts = frame_number / self.fps if self.fps > 0 else 0
        hud = f"Frame:{frame_number} | T:{ts:.2f}s | Ann.FPS:{fps_display:.0f}"
        cv2.putText(
            frame, hud,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
        )
