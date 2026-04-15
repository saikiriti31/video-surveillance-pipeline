"""
ZoneManager — loads and manages polygon zones from a JSON config.

Zone JSON format:
{
  "zones": [
    {
      "id": "zone_entrance",
      "label": "Entrance",
      "color": [0, 0, 255],
      "polygon": [[100, 200], [400, 200], [400, 500], [100, 500]],
      "loitering_threshold_seconds": 10.0,
      "alert_cooldown_seconds": 30.0
    }
  ]
}

Uses Shapely for robust point-in-polygon:
- Handles convex AND concave polygons correctly
- Handles edge/vertex touching cases
- Much faster than naive ray casting for many points
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Shapely is required; graceful ImportError message
try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not installed — falling back to ray-casting PIP")


@dataclass
class Zone:
    """Represents a single surveillance zone with its configuration."""
    id: str
    label: str
    polygon: List[Tuple[int, int]]          # list of (x, y) vertex coords
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR for OpenCV drawing
    loitering_threshold_seconds: float = 10.0
    alert_cooldown_seconds: float = 30.0

    def __post_init__(self):
        self.color = tuple(self.color)
        self.polygon = [tuple(pt) for pt in self.polygon]
        if SHAPELY_AVAILABLE:
            self._shapely_poly = ShapelyPolygon(self.polygon)
            if not self._shapely_poly.is_valid:
                # Auto-repair with buffer(0) trick
                self._shapely_poly = self._shapely_poly.buffer(0)
                logger.warning(f"Zone '{self.id}' polygon was invalid; auto-repaired")
        else:
            self._shapely_poly = None

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside the zone polygon."""
        if SHAPELY_AVAILABLE and self._shapely_poly:
            return self._shapely_poly.contains(Point(x, y))
        return self._ray_cast_pip(x, y)

    def _ray_cast_pip(self, px: float, py: float) -> bool:
        """
        Fallback ray-casting point-in-polygon.
        O(n) where n = number of polygon vertices.
        """
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside

    def numpy_polygon(self):
        """Return polygon as numpy int32 array for cv2.polylines/fillPoly."""
        import numpy as np
        return np.array(self.polygon, dtype=np.int32)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "polygon": self.polygon,
            "color": list(self.color),
            "loitering_threshold_seconds": self.loitering_threshold_seconds,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
        }


class ZoneManager:
    """Loads zones from JSON and provides spatial query methods."""

    def __init__(self, zones_config_path: str):
        self.config_path = Path(zones_config_path)
        self.zones: List[Zone] = []
        self._load_zones()

    def _load_zones(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Zone config not found: {self.config_path}")

        with open(self.config_path) as f:
            data = json.load(f)

        raw_zones = data.get("zones", [])
        if not raw_zones:
            logger.warning(f"No zones defined in {self.config_path}")

        for zdata in raw_zones:
            try:
                zone = Zone(
                    id=zdata["id"],
                    label=zdata.get("label", zdata["id"]),
                    polygon=zdata["polygon"],
                    color=zdata.get("color", [0, 0, 255]),
                    loitering_threshold_seconds=zdata.get("loitering_threshold_seconds", 10.0),
                    alert_cooldown_seconds=zdata.get("alert_cooldown_seconds", 30.0),
                )
                self.zones.append(zone)
                logger.info(f"Loaded zone: {zone.id} ({len(zone.polygon)} vertices)")
            except (KeyError, TypeError) as exc:
                logger.error(f"Skipping invalid zone definition: {exc}")

        logger.info(f"ZoneManager ready with {len(self.zones)} zones")

    def get_zones_for_point(self, x: float, y: float) -> List[Zone]:
        """Return all zones that contain point (x, y)."""
        return [z for z in self.zones if z.contains_point(x, y)]

    def get_zone_by_id(self, zone_id: str) -> Optional[Zone]:
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def reload(self):
        """Hot-reload zones config without restarting pipeline."""
        self.zones = []
        self._load_zones()
