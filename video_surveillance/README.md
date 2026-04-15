# Video Surveillance Pipeline
### Detection, Tracking & Event Recognition

A production-ready computer vision pipeline for security camera footage. Detects people, tracks them persistently across frames, and identifies zone-based events (intrusion, loitering) with structured event logging and annotated video output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SurveillancePipeline                        │
│                      (src/pipeline.py)                          │
└──────────┬──────────────────────────────────────────────────────┘
           │ frames (one at a time, never accumulated)
           ▼
┌──────────────────────┐
│   PersonDetector     │  YOLOv9{t/s/m/c/e} + ByteTrack
│  (src/detector/)     │  • COCO class 0 only (person)
│                      │  • FP16 on CUDA, FP32 on CPU
│  Output: List[Detection]  (track_id, bbox, confidence, ts)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    ZoneManager       │  Shapely polygons (+ ray-cast fallback)
│   (src/zones/)       │  • Loaded from zones.json
│                      │  • Concave polygon support
│  Query: get_zones_for_point(foot_x, foot_y)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    EventEngine       │  Per (track_id, zone_id) state machine
│   (src/events/)      │  • ZONE_INTRUSION on entry
│                      │  • LOITERING after threshold + velocity check
│                      │  • ZONE_EXIT tracking
│                      │  • Occlusion grace period
│  Output: List[Event]
└────────┬──────┬──────┘
         │      │
         ▼      ▼
┌──────────┐  ┌────────────────┐
│ Video    │  │  EventLogger   │
│ Writer   │  │ (src/output/)  │
│          │  │                │
│ Annotated│  │ events.json    │
│ MP4 out  │  │ events.csv     │
└──────────┘  └────────────────┘
```

### Pipeline Stages

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **Detection** | `src/detector/detector.py` | YOLOv9 inference, person class filter, confidence gate |
| **Tracking** | ByteTrack (via Ultralytics) | Assign persistent IDs, re-ID on re-entry via low-conf buffer |
| **Zone Spatial** | `src/zones/zone_manager.py` | Load polygons, point-in-polygon, zone config |
| **Event Logic** | `src/events/event_engine.py` | State machine, loitering clock, cooldowns, occlusion grace |
| **Output** | `src/output/` | Annotated video, JSON/CSV event logs |
| **Orchestration** | `src/pipeline.py` | Frame loop, frame sampling, memory management |
| **CLI** | `run.py` | Argument parsing, multi-video support |

---

## Model Selection

### Detector: YOLOv9 (Ultralytics)

**Chosen:** YOLOv9t (default) / YOLOv9m (GPU recommended)

**Why YOLOv9 over alternatives:**

| Model | Speed (V100) | mAP@0.5 | Notes |
|-------|-------------|---------|-------|
| YOLOv9t | ~200 FPS | 52.9 | Best CPU/edge option |
| YOLOv9m | ~100 FPS | 63.9 | Best general-purpose GPU |
| Faster R-CNN | ~10 FPS | 60+ | High accuracy, too slow for real-time |
| YOLOv5s | ~120 FPS | 56.0 | Older, YOLOv9 supersedes it |
| RT-DETR | ~60 FPS | 67+ | Transformer-based, high memory |

**Key reasons:**
- ONNX exportable for edge deployment
- Native ByteTrack integration (no glue code)
- Pre-trained on COCO — person class 0 is high quality
- Runs on CPU without CUDA drivers (critical for diverse deployments)
- `model.track(persist=True)` keeps tracker state across frames automatically

### Tracker: ByteTrack (integrated in Ultralytics)

**Why ByteTrack over DeepSORT:**

| Tracker | Re-ID Model Required | Occlusion Handling | Speed |
|---------|---------------------|--------------------|-------|
| **ByteTrack** | ❌ No | ✅ Low-conf buffer | ~2ms/frame |
| DeepSORT | ✅ Yes (extra model) | ⚠️ Moderate | ~15ms/frame |
| StrongSORT | ✅ Yes | ✅ Good | ~20ms/frame |
| SORT | ❌ No | ❌ Poor | ~1ms/frame |

ByteTrack's core insight: keep low-confidence detections in a buffer rather than discarding them. When a track disappears (occlusion) and reappears, the low-confidence buffer catches the re-entry and assigns the **same ID** — eliminating the need for a separate Re-ID model in most CCTV scenarios.

---

## Setup

### Option 1: Local Python Environment

```bash
# Clone / extract the repo
cd video_surveillance

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# YOLOv9 weights auto-download on first run
# Or pre-download manually:
python -c "from ultralytics import YOLO; YOLO('yolov9t.pt'); YOLO('yolov9m.pt')"
```

---

## Usage

### Basic Run
```bash
python run.py --video input.mp4 --zones configs/zones.json --output results/
```

### Full Options
```bash
python run.py \
  --video clip1.mp4 clip2.mp4 \     # Multiple videos
  --zones configs/zones.json \
  --output results/ \
  --model m \                        # yolov9m for better accuracy
  --confidence 0.35 \               # Detection threshold
  --device cuda \                   # Force GPU
  --target-fps 15 \                 # Downsample for speed
  --grace-frames 20 \               # Occlusion tolerance
  --log-level DEBUG
```

### Without Annotated Video (Events Only — Much Faster)
```bash
python run.py --video input.mp4 --zones configs/zones.json --no-video
```

### Inspect Video Metadata
```bash
python run.py --video input.mp4 --zones configs/zones.json --video-info
```

### List Zones
```bash
python run.py --video input.mp4 --zones configs/zones.json --list-zones
```

---

## Zone Configuration

Zones are defined in JSON. Each zone is a polygon (list of [x, y] pixel coordinates).

```json
{
  "zones": [
    {
      "id": "zone_restricted_entrance",
      "label": "Restricted Entrance",
      "color": [0, 0, 255],
      "polygon": [
        [50, 200], [350, 200], [350, 500], [50, 500]
      ],
      "loitering_threshold_seconds": 10.0,
      "alert_cooldown_seconds": 30.0
    }
  ]
}
```

| Field | Description | Default |
|-------|-------------|---------|
| `id` | Unique zone identifier | required |
| `label` | Human-readable name shown on video | required |
| `color` | BGR color for overlay `[B, G, R]` | `[0, 0, 255]` |
| `polygon` | List of `[x, y]` pixel vertices (min 3) | required |
| `loitering_threshold_seconds` | Time before loitering alert fires | `10.0` |
| `alert_cooldown_seconds` | Min gap between repeated alerts | `30.0` |

**Tip:** Use a tool like [CVAT](https://cvat.ai/) or draw polygons on a frame screenshot to get accurate pixel coordinates for your camera.

---

## Output Files

After processing `input.mp4`, the `results/` directory will contain:

```
results/
├── input_annotated.mp4          # Annotated video with overlays
├── input_events.json            # Full structured event log
└── input_events.csv             # Flat CSV for spreadsheet analysis
```

### Event JSON Structure
```json
{
  "video": "input",
  "total_events": 12,
  "summary": {
    "intrusions": 8,
    "loiterings": 4,
    "deduplicated": 3
  },
  "events": [
    {
      "event_type": "zone_intrusion",
      "zone_id": "zone_restricted_entrance",
      "zone_label": "Restricted Entrance",
      "track_id": 3,
      "frame_number": 145,
      "timestamp": 4.8333,
      "bbox": [230, 180, 310, 390],
      "confidence": 0.872,
      "duration_seconds": null
    },
    {
      "event_type": "loitering",
      "zone_id": "zone_parking_lot",
      "zone_label": "Parking Lot",
      "track_id": 7,
      "frame_number": 920,
      "timestamp": 30.667,
      "bbox": [410, 120, 490, 340],
      "confidence": 0.904,
      "duration_seconds": 18.4
    }
  ]
}
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ -v --cov=src --cov-report=html
```

Test coverage:
- `tests/test_zones.py` — ZoneManager loading, point-in-polygon (square, L-shape, boundary, concave)
- `tests/test_events.py` — Intrusion fire/cooldown, loitering trigger/reset, occlusion grace, multi-track, edge cases

---

## Performance Notes

### Measured on Intel i7-12700 (CPU only, yolov9t, 1080p)

| Resolution | Model | FPS (inference) | FPS (pipeline) |
|-----------|-------|----------------|----------------|
| 1920×1080 | yolov9t | ~8–12 | ~6–10 |
| 1280×720 | yolov9t | ~18–25 | ~15–20 |
| 640×480 | yolov9t | ~35–45 | ~28–38 |

### GPU (RTX 3080, yolov9m, 1080p)
| Resolution | Model | FPS (inference) | FPS (pipeline) |
|-----------|-------|----------------|----------------|
| 1920×1080 | yolov9m | ~55–70 | ~45–60 |
| 1920×1080 | yolov9t | ~120–160 | ~90–130 |

**Memory usage:**
- YOLOv9t: ~200MB GPU VRAM
- YOLOv9m: ~450MB GPU VRAM
- Long videos (1+ hour): constant ~300MB RAM (frame-by-frame processing, no accumulation)

### Speed Optimization Tips
1. `--target-fps 10` — process at 10fps; most events still detected, ~3x speedup
2. `--model t` — tiny model is fastest
3. `--no-video` — skip annotation rendering (~20% faster)
4. `--device cuda` — GPU gives 5–10x speedup on full-res video

---

## Recommended Datasets

| Dataset | Best For | Note |
|---------|----------|------|
| [MOT17](https://motchallenge.net/data/MOT17/) | Tracking accuracy, clean ground truth | Start here |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | Real CCTV, loitering/fighting scenarios | Messier, more realistic |
| [VIRAT](https://viratdata.org/) | Outdoor pedestrian scenes | High quality |


## Known Limitations & Future Improvements

### Current Limitations

| Area | Limitation |
|------|-----------|
| **Re-ID** | ByteTrack re-ID relies on spatial proximity. Cross-camera re-ID not supported. |
| **Loitering** | Velocity check is per-frame; fast oscillation in place may reset the clock incorrectly |
| **Crowd scenes** | ID switches increase when >30 people tightly packed |
| **Camera shake** | Severe shake causes zone polygon misalignment (zones are static pixel coords) |
| **Night scenes** | YOLOv9 pre-trained on daylight COCO; accuracy drops in IR/very-dark footage |
| **Zone coords** | Pixel-absolute; zones must be re-drawn if camera angle changes |

### With More Time

1. **Perspective-aware zones** — homography transform to world-space coordinates; zones defined in meters
2. **Cross-camera Re-ID** — add OSNet/TransReID appearance features for multi-camera tracking
3. **Fine-tuned detection** — fine-tune YOLOv9 on UCF-Crime / VIRAT for domain-specific improvements
4. **WebSocket live dashboard** — FastAPI + WebSocket to stream annotated MJPEG frames + events
5. **MOTA/MOTP evaluation** — evaluation script against MOT17 ground truth
6. **Alert deduplication** — smarter dedup using embedding similarity, not just time window
7. **GPU-accelerated annotation** — use CUDA-based OpenCV for annotation rendering bottleneck

---

## Project Structure

```
video_surveillance/
├── run.py                      # CLI entrypoint
├── requirements.txt
├── README.md
├── configs/
│   ├── zones.json              # Default zone config
│   └── zones_mot17.json        # MOT17-adapted zones
├── src/
│   ├── pipeline.py             # Main orchestrator
│   ├── detector/
│   │   └── detector.py         # YOLOv9 + ByteTrack wrapper
│   ├── zones/
│   │   └── zone_manager.py     # Zone loading + point-in-polygon
│   ├── events/
│   │   └── event_engine.py     # State machine, intrusion/loitering
│   ├── output/
│   │   ├── video_writer.py     # Annotated video output
│   │   └── event_logger.py     # JSON/CSV event log
│   └── utils/
│       └── pipeline_utils.py   # VideoInfo, FrameSampler, PipelineStats
├── tests/
│   ├── test_zones.py
│   └── test_events.py
└── results/                    # Output files (gitignored)
```
