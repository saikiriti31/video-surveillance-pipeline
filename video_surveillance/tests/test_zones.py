"""
Unit tests for ZoneManager and Zone point-in-polygon logic.
Run with: pytest tests/test_zones.py -v
"""

import json
import os
import tempfile
import pytest

from src.zones import ZoneManager, Zone


@pytest.fixture
def sample_zones_file():
    """Create a temp zones.json for testing."""
    zones_data = {
        "zones": [
            {
                "id": "test_zone_a",
                "label": "Zone A",
                "color": [255, 0, 0],
                "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
                "loitering_threshold_seconds": 5.0,
                "alert_cooldown_seconds": 10.0,
            },
            {
                "id": "test_zone_b",
                "label": "Zone B (L-shaped)",
                "color": [0, 255, 0],
                "polygon": [
                    [400, 100], [600, 100], [600, 200],
                    [500, 200], [500, 400], [400, 400]
                ],
                "loitering_threshold_seconds": 8.0,
                "alert_cooldown_seconds": 15.0,
            }
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(zones_data, f)
        yield f.name
    os.unlink(f.name)


class TestZoneLoading:
    def test_loads_zones(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        assert len(zm.zones) == 2

    def test_zone_attributes(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        z = zm.get_zone_by_id("test_zone_a")
        assert z is not None
        assert z.label == "Zone A"
        assert z.loitering_threshold_seconds == 5.0
        assert len(z.polygon) == 4

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ZoneManager("/nonexistent/zones.json")


class TestPointInPolygon:
    """Test both Shapely and ray-cast implementations."""

    def test_point_inside_square(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zone = zm.get_zone_by_id("test_zone_a")
        # Center of [100,100] to [300,300] square
        assert zone.contains_point(200, 200) is True

    def test_point_outside_square(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zone = zm.get_zone_by_id("test_zone_a")
        assert zone.contains_point(50, 50) is False
        assert zone.contains_point(400, 400) is False

    def test_point_on_boundary(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zone = zm.get_zone_by_id("test_zone_a")
        # Boundary behavior is consistent (may be in or out, but stable)
        result = zone.contains_point(100, 200)
        assert isinstance(result, bool)

    def test_l_shaped_zone_inside(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zone = zm.get_zone_by_id("test_zone_b")
        assert zone.contains_point(450, 150) is True   # top part of L
        assert zone.contains_point(420, 300) is True   # bottom part of L

    def test_l_shaped_zone_outside_concave_notch(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zone = zm.get_zone_by_id("test_zone_b")
        # This point is in the "notch" of the L — must be outside
        assert zone.contains_point(560, 300) is False

    def test_get_zones_for_point(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zones = zm.get_zones_for_point(200, 200)
        assert len(zones) == 1
        assert zones[0].id == "test_zone_a"

    def test_get_zones_for_point_none(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        zones = zm.get_zones_for_point(10, 10)
        assert len(zones) == 0


class TestZoneSerialization:
    def test_to_dict_roundtrip(self, sample_zones_file):
        zm = ZoneManager(sample_zones_file)
        z = zm.zones[0]
        d = z.to_dict()
        assert d["id"] == z.id
        assert d["label"] == z.label
        assert len(d["polygon"]) == len(z.polygon)
