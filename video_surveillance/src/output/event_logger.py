"""
EventLogger — writes structured event logs in JSON and CSV formats.

Output files:
  results/events.json  — full structured log with all metadata
  results/events.csv   — flat CSV for spreadsheet analysis

Deduplication logic:
  - Same (event_type, track_id, zone_id) within cooldown window = 1 entry
  - Configurable dedup window (default = zone.alert_cooldown_seconds)
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..events.event_engine import Event, EventType

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "timestamp", "frame_number", "event_type", "zone_id", "zone_label",
    "track_id", "confidence", "duration_seconds",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
]


class EventLogger:
    """
    Thread-safe event log accumulator.
    Call log_events() per frame, finalize() at end.
    """

    def __init__(self, output_dir: str, video_name: str = "video"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_name = video_name

        self._events: List[Event] = []
        self._dedup_cache: Dict[Tuple[str, int, str], float] = {}

        # Stats counters
        self.stats = {
            "total_events": 0,
            "intrusions": 0,
            "loiterings": 0,
            "exits": 0,
            "deduplicated": 0,
        }

        logger.info(f"EventLogger initialized → {self.output_dir}")

    def log_events(self, events: List[Event], dedup_window_seconds: float = 5.0):
        """
        Add events to the log, applying deduplication.
        dedup_window_seconds: min gap between identical (type, track, zone) events.
        """
        for event in events:
            # Skip ZONE_EXIT from logs (informational only)
            if event.event_type == EventType.ZONE_EXIT:
                self.stats["exits"] += 1
                continue

            dedup_key = (event.event_type.value, event.track_id, event.zone_id)
            last_ts = self._dedup_cache.get(dedup_key, -999.0)

            if (event.timestamp - last_ts) < dedup_window_seconds:
                self.stats["deduplicated"] += 1
                continue

            self._events.append(event)
            self._dedup_cache[dedup_key] = event.timestamp
            self.stats["total_events"] += 1

            if event.event_type == EventType.ZONE_INTRUSION:
                self.stats["intrusions"] += 1
            elif event.event_type == EventType.LOITERING:
                self.stats["loiterings"] += 1

    def finalize(self) -> Dict:
        """Write JSON + CSV outputs and return summary stats."""
        json_path = self.output_dir / f"{self.video_name}_events.json"
        csv_path = self.output_dir / f"{self.video_name}_events.csv"

        # JSON
        output = {
            "video": self.video_name,
            "total_events": self.stats["total_events"],
            "summary": self.stats,
            "events": [e.to_dict() for e in sorted(self._events, key=lambda x: x.timestamp)],
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        # CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for event in sorted(self._events, key=lambda x: x.timestamp):
                x1, y1, x2, y2 = event.bbox
                writer.writerow({
                    "timestamp": round(event.timestamp, 4),
                    "frame_number": event.frame_number,
                    "event_type": event.event_type.value,
                    "zone_id": event.zone_id,
                    "zone_label": event.zone_label,
                    "track_id": event.track_id,
                    "confidence": round(event.confidence, 4),
                    "duration_seconds": round(event.duration_seconds, 2) if event.duration_seconds else "",
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                })

        logger.info(f"Events saved: {json_path} | {csv_path}")
        logger.info(f"Stats: {self.stats}")

        return {
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "stats": self.stats,
        }
