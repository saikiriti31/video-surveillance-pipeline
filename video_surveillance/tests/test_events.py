"""
Unit tests for EventEngine state machine.
Run with: pytest tests/test_events.py -v
"""

import json
import os
import tempfile
import pytest

from src.detector.detector import Detection
from src.zones import ZoneManager, Zone
from src.events import EventEngine, Event
from src.events.event_engine import EventType


@pytest.fixture
def simple_zones_file():
    """A single restricted zone at [0,0] to [500,500]."""
    data = {
        "zones": [
            {
                "id": "zone_test",
                "label": "Test Zone",
                "color": [255, 0, 0],
                "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
                "loitering_threshold_seconds": 5.0,
                "alert_cooldown_seconds": 3.0,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        yield f.name
    os.unlink(f.name)


def make_detection(track_id, x1, y1, x2, y2, frame_number, fps=25.0, conf=0.9):
    return Detection(
        track_id=track_id,
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        frame_number=frame_number,
        timestamp=frame_number / fps,
    )


class TestIntrusionDetection:
    def test_intrusion_fires_on_entry(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        engine = EventEngine(zm, fps=25.0)

        # Person at center of zone (foot point = (250, 400) — well inside)
        det = make_detection(1, 200, 200, 300, 400, frame_number=10)
        events = engine.process_frame([det], frame_number=10)

        intrusions = [e for e in events if e.event_type == EventType.ZONE_INTRUSION]
        assert len(intrusions) == 1
        assert intrusions[0].track_id == 1
        assert intrusions[0].zone_id == "zone_test"

    def test_intrusion_cooldown_prevents_duplicate(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        engine = EventEngine(zm, fps=25.0)

        all_events = []
        # Process 10 consecutive frames (all within cooldown)
        for fn in range(1, 11):
            det = make_detection(1, 200, 200, 300, 400, frame_number=fn)
            events = engine.process_frame([det], frame_number=fn)
            all_events.extend(events)

        intrusions = [e for e in all_events if e.event_type == EventType.ZONE_INTRUSION]
        assert len(intrusions) == 1  # Only one alert within cooldown window

    def test_no_intrusion_when_outside_zone(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        engine = EventEngine(zm, fps=25.0)

        # Foot point at (800, 900) — outside zone
        det = make_detection(1, 750, 700, 850, 900, frame_number=5)
        events = engine.process_frame([det], frame_number=5)

        intrusions = [e for e in events if e.event_type == EventType.ZONE_INTRUSION]
        assert len(intrusions) == 0

    def test_multiple_tracks_independent(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        engine = EventEngine(zm, fps=25.0)

        dets = [
            make_detection(1, 50, 50, 150, 150, frame_number=1),
            make_detection(2, 200, 200, 300, 300, frame_number=1),
            make_detection(3, 350, 350, 450, 450, frame_number=1),
        ]
        events = engine.process_frame(dets, frame_number=1)
        intrusions = [e for e in events if e.event_type == EventType.ZONE_INTRUSION]
        assert len(intrusions) == 3
        track_ids = {e.track_id for e in intrusions}
        assert track_ids == {1, 2, 3}


class TestLoiteringDetection:
    def test_loitering_fires_after_threshold(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        fps = 25.0
        engine = EventEngine(zm, fps=fps, stationary_threshold_px=100.0)

        # threshold = 5.0s @ 25fps = 125 frames
        # Run 130 frames with person stationary in zone
        all_events = []
        for fn in range(0, 130):
            det = make_detection(1, 200, 200, 300, 400, frame_number=fn, fps=fps)
            events = engine.process_frame([det], frame_number=fn)
            all_events.extend(events)

        loiterings = [e for e in all_events if e.event_type == EventType.LOITERING]
        assert len(loiterings) >= 1
        assert loiterings[0].duration_seconds >= 5.0

    def test_loitering_resets_when_moving(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        fps = 25.0
        engine = EventEngine(zm, fps=fps, stationary_threshold_px=5.0)  # tight threshold

        all_events = []
        for fn in range(0, 200):
            # Person moves 50px every frame — should not loiter
            x = 50 + (fn % 400)
            if x > 400:
                x = 400
            det = make_detection(1, x, 50, x + 100, 250, frame_number=fn, fps=fps)
            events = engine.process_frame([det], frame_number=fn)
            all_events.extend(events)

        loiterings = [e for e in all_events if e.event_type == EventType.LOITERING]
        assert len(loiterings) == 0


class TestEdgeCases:
    def test_empty_frame(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        engine = EventEngine(zm, fps=25.0)
        events = engine.process_frame([], frame_number=1)
        assert events == []

    def test_track_reentry_after_occlusion(self, simple_zones_file):
        """Track disappears for < grace_frames then reappears — state preserved."""
        zm = ZoneManager(simple_zones_file)
        fps = 25.0
        engine = EventEngine(zm, fps=fps, grace_frames=30)

        all_events = []
        # Frames 0-9: visible
        for fn in range(10):
            det = make_detection(1, 200, 200, 300, 400, frame_number=fn, fps=fps)
            events = engine.process_frame([det], frame_number=fn)
            all_events.extend(events)

        # Frames 10-19: occluded (10 < 30 grace frames)
        for fn in range(10, 20):
            events = engine.process_frame([], frame_number=fn)
            all_events.extend(events)

        # Frame 20: reappears — should NOT re-fire intrusion (still within cooldown)
        det = make_detection(1, 200, 200, 300, 400, frame_number=20, fps=fps)
        events = engine.process_frame([det], frame_number=20)
        all_events.extend(events)

        intrusions = [e for e in all_events if e.event_type == EventType.ZONE_INTRUSION]
        # Should have exactly 1 intrusion (on first entry), not 2
        assert len(intrusions) == 1

    def test_zone_exit_fires_on_leave(self, simple_zones_file):
        zm = ZoneManager(simple_zones_file)
        fps = 25.0
        engine = EventEngine(zm, fps=fps)

        # Frame 1: inside zone
        det1 = make_detection(1, 200, 200, 300, 400, frame_number=1, fps=fps)
        engine.process_frame([det1], frame_number=1)

        # Frame 2: outside zone (foot at 800, 900)
        det2 = make_detection(1, 750, 700, 850, 900, frame_number=2, fps=fps)
        events = engine.process_frame([det2], frame_number=2)

        exits = [e for e in events if e.event_type == EventType.ZONE_EXIT]
        assert len(exits) == 1
        assert exits[0].track_id == 1
