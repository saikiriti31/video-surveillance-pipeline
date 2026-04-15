"""
EventEngine — Frame-level state machine for zone-based event detection.

Event Types:
  1. ZONE_INTRUSION  — person's foot-point enters a defined zone
  2. LOITERING       — person remains in zone beyond loitering_threshold_seconds

State management:
  - Per-(track_id, zone_id) state tracked across frames
  - Cooldown prevents duplicate alerts for the same event
  - Velocity-based "stationary" check for loitering (displacement < threshold)
  - Handles track ID switches gracefully (new ID = fresh state)

Edge cases handled:
  - Occlusion (track disappears for < grace_frames): state preserved
  - ID switch: old state expires after grace_frames, new ID gets fresh state
  - Empty frames: no state mutation
  - Crowded scenes: each (track_id, zone_id) pair tracked independently
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..detector.detector import Detection
from ..zones.zone_manager import Zone, ZoneManager

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    ZONE_INTRUSION = "zone_intrusion"
    LOITERING = "loitering"
    ZONE_EXIT = "zone_exit"           # informational — person left zone


@dataclass
class Event:
    """A detected surveillance event."""
    event_type: EventType
    zone_id: str
    zone_label: str
    track_id: int
    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    duration_seconds: Optional[float] = None   # for loitering events
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "track_id": self.track_id,
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 4),
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "duration_seconds": round(self.duration_seconds, 2) if self.duration_seconds else None,
            "metadata": self.metadata,
        }


@dataclass
class TrackZoneState:
    """Per (track_id, zone_id) state maintained across frames."""
    track_id: int
    zone_id: str
    entry_timestamp: float        # when person first entered zone
    entry_frame: int
    last_seen_frame: int          # for occlusion grace period
    last_position: Optional[Tuple[float, float]] = None

    # Cooldown tracking
    last_intrusion_alert_ts: float = -999.0
    last_loitering_alert_ts: float = -999.0
    loitering_triggered: bool = False

    def time_in_zone(self, current_ts: float) -> float:
        return current_ts - self.entry_timestamp


class EventEngine:
    """
    Processes detections + zones per frame to emit Events.

    Parameters
    ----------
    zone_manager : ZoneManager
    fps : float
        Video FPS — used to compute duration from frame counts.
    grace_frames : int
        How many consecutive frames without detection before a track
        is considered gone (not just occluded). Default 15 (~0.5s at 30fps).
    stationary_threshold_px : float
        Max pixel displacement between frames to be considered "stationary"
        for loitering detection. Default 15px.
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        fps: float,
        grace_frames: int = 15,
        stationary_threshold_px: float = 15.0,
    ):
        self.zone_manager = zone_manager
        self.fps = fps
        self.grace_frames = grace_frames
        self.stationary_threshold_px = stationary_threshold_px

        # Key: (track_id, zone_id) → TrackZoneState
        self._active_states: Dict[Tuple[int, str], TrackZoneState] = {}
        # Tracks currently visible this frame
        self._visible_track_ids: Set[int] = set()

    def process_frame(
        self,
        detections: List[Detection],
        frame_number: int,
    ) -> List[Event]:
        """
        Main entry point — call once per frame.
        Returns list of Events triggered this frame (may be empty).
        """
        events: List[Event] = []
        current_track_ids = {d.track_id for d in detections}

        # --- For each detection, check zone membership ---
        for det in detections:
            foot_x, foot_y = det.foot_point
            zones_hit = self.zone_manager.get_zones_for_point(foot_x, foot_y)

            active_zone_ids = {z.id for z in zones_hit}

            for zone in zones_hit:
                key = (det.track_id, zone.id)

                if key not in self._active_states:
                    # New zone entry
                    state = TrackZoneState(
                        track_id=det.track_id,
                        zone_id=zone.id,
                        entry_timestamp=det.timestamp,
                        entry_frame=frame_number,
                        last_seen_frame=frame_number,
                        last_position=det.foot_point,
                    )
                    self._active_states[key] = state

                    # Fire ZONE_INTRUSION if cooldown elapsed
                    intrusion_event = self._try_intrusion_alert(det, zone, state)
                    if intrusion_event:
                        events.append(intrusion_event)
                else:
                    state = self._active_states[key]
                    state.last_seen_frame = frame_number

                    # Check LOITERING
                    loitering_event = self._check_loitering(det, zone, state, frame_number)
                    if loitering_event:
                        events.append(loitering_event)

                    state.last_position = det.foot_point

            # Handle ZONE_EXIT for zones this track has left
            for key in list(self._active_states.keys()):
                tid, zid = key
                if tid == det.track_id and zid not in active_zone_ids:
                    state = self._active_states.pop(key)
                    exit_event = Event(
                        event_type=EventType.ZONE_EXIT,
                        zone_id=zid,
                        zone_label=self.zone_manager.get_zone_by_id(zid).label if self.zone_manager.get_zone_by_id(zid) else zid,
                        track_id=det.track_id,
                        frame_number=frame_number,
                        timestamp=det.timestamp,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        duration_seconds=state.time_in_zone(det.timestamp),
                    )
                    events.append(exit_event)
                    logger.debug(f"Track {det.track_id} exited zone {zid} after {exit_event.duration_seconds:.1f}s")

        # --- Expire states for tracks no longer visible (beyond grace period) ---
        self._expire_old_states(current_track_ids, frame_number)

        self._visible_track_ids = current_track_ids
        return events

    def reset(self):
        """Clear all state — call between videos."""
        self._active_states.clear()
        self._visible_track_ids.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_intrusion_alert(
        self, det: Detection, zone: Zone, state: TrackZoneState
    ) -> Optional[Event]:
        elapsed_since_last = det.timestamp - state.last_intrusion_alert_ts
        if elapsed_since_last < zone.alert_cooldown_seconds:
            return None  # cooldown active

        state.last_intrusion_alert_ts = det.timestamp
        logger.info(
            f"INTRUSION | track={det.track_id} zone={zone.id} t={det.timestamp:.2f}s"
        )
        return Event(
            event_type=EventType.ZONE_INTRUSION,
            zone_id=zone.id,
            zone_label=zone.label,
            track_id=det.track_id,
            frame_number=det.frame_number,
            timestamp=det.timestamp,
            bbox=det.bbox,
            confidence=det.confidence,
        )

    def _check_loitering(
        self,
        det: Detection,
        zone: Zone,
        state: TrackZoneState,
        frame_number: int,
    ) -> Optional[Event]:
        time_in_zone = state.time_in_zone(det.timestamp)

        if time_in_zone < zone.loitering_threshold_seconds:
            return None

        # Check cooldown
        elapsed_since_last = det.timestamp - state.last_loitering_alert_ts
        if elapsed_since_last < zone.alert_cooldown_seconds:
            return None

        # Velocity check — ensure person is reasonably stationary
        if state.last_position:
            dx = det.foot_point[0] - state.last_position[0]
            dy = det.foot_point[1] - state.last_position[1]
            displacement = math.sqrt(dx * dx + dy * dy)
            # Scale threshold by FPS (displacement is per-frame)
            if displacement > self.stationary_threshold_px:
                # Person is moving — reset loitering clock
                state.entry_timestamp = det.timestamp
                state.loitering_triggered = False
                return None

        state.last_loitering_alert_ts = det.timestamp
        state.loitering_triggered = True

        logger.info(
            f"LOITERING | track={det.track_id} zone={zone.id} "
            f"duration={time_in_zone:.1f}s t={det.timestamp:.2f}s"
        )
        return Event(
            event_type=EventType.LOITERING,
            zone_id=zone.id,
            zone_label=zone.label,
            track_id=det.track_id,
            frame_number=det.frame_number,
            timestamp=det.timestamp,
            bbox=det.bbox,
            confidence=det.confidence,
            duration_seconds=time_in_zone,
        )

    def _expire_old_states(
        self, visible_track_ids: Set[int], current_frame: int
    ):
        """Remove states for tracks gone beyond grace period."""
        expired_keys = []
        for key, state in self._active_states.items():
            tid, _ = key
            if tid not in visible_track_ids:
                frames_absent = current_frame - state.last_seen_frame
                if frames_absent > self.grace_frames:
                    expired_keys.append(key)

        for key in expired_keys:
            del self._active_states[key]
            logger.debug(f"Expired state for track/zone: {key}")
