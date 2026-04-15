#!/usr/bin/env python3
"""
Video Surveillance Pipeline — CLI Entrypoint

Usage:
    # Single video
    python run.py --video input.mp4 --zones configs/zones.json --output results/

    # Multiple videos
    python run.py --video v1.mp4 v2.mp4 --zones configs/zones.json --output results/

    # GPU, higher accuracy model, no video output
    python run.py --video input.mp4 --zones zones.json --model m --device cuda --no-video

    # Downsample to 10fps for faster processing
    python run.py --video input.mp4 --zones zones.json --target-fps 10

    # Verbose logging
    python run.py --video input.mp4 --zones zones.json --log-level DEBUG
"""

import argparse
import json
import logging
import sys
from pathlib import Path


def setup_logging(level: str):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Surveillance: Detection, Tracking & Event Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--video", nargs="+", required=True,
        help="Path(s) to input video file(s)"
    )
    parser.add_argument(
        "--zones", required=True,
        help="Path to zones.json config file"
    )

    # Output
    parser.add_argument(
        "--output", default="results/",
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip annotated video output (faster — events only)"
    )

    # Model
    parser.add_argument(
        "--model", default="t", choices=["t", "s", "m", "c", "e"],
        help=(
            "YOLOv9 model size (default: t):\n"
            "  t = tiny    — fastest, works on CPU, good for testing\n"
            "  s = small   — balanced speed and accuracy\n"
            "  m = medium  — best for GPU, recommended for production\n"
            "  c = compact — high accuracy\n"
            "  e = extended— most accurate, needs strong GPU"
        )
    )
    parser.add_argument(
        "--confidence", type=float, default=0.2,
        help="Detection confidence threshold 0.0-1.0 (default: 0.2). Lower = catches more people (umbrella/occlusion)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:0', 'mps'. Auto-detected if omitted."
    )

    # Enhancement flags
    parser.add_argument(
        "--enhance", action="store_true",
        help="Enable frame enhancement (CLAHE + bilateral filter + brightness boost). "
             "Use for night scenes, fog, rain, or low-quality CCTV footage."
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable Test Time Augmentation (TTA). Detects partially hidden people "
             "(umbrella, occlusion, unusual angles). Slower but more accurate."
    )

    # Performance
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help="Downsample to this FPS for processing (e.g. 10). None = process all frames."
    )
    parser.add_argument(
        "--grace-frames", type=int, default=15,
        help="Frames of occlusion grace period before track state expires (default: 15)"
    )

    # Logging
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )

    # Info modes
    parser.add_argument(
        "--list-zones", action="store_true",
        help="Print loaded zones and exit"
    )
    parser.add_argument(
        "--video-info", action="store_true",
        help="Print video metadata and exit (no processing)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run")

    # Validate inputs
    for vp in args.video:
        if not Path(vp).exists():
            logger.error(f"Video file not found: {vp}")
            sys.exit(1)

    if not Path(args.zones).exists():
        logger.error(f"Zones config not found: {args.zones}")
        sys.exit(1)

    # Info modes
    if args.video_info:
        from src.utils import get_video_info
        for vp in args.video:
            info = get_video_info(vp)
            print(json.dumps(info.__dict__, indent=2))
        return

    if args.list_zones:
        from src.zones import ZoneManager
        zm = ZoneManager(args.zones)
        for z in zm.zones:
            print(json.dumps(z.to_dict(), indent=2))
        return

    # Import pipeline (deferred to avoid slow startup when just listing info)
    from src.pipeline import SurveillancePipeline

    # Log active enhancement settings
    if args.enhance:
        logger.info("Frame enhancement ENABLED (CLAHE + bilateral filter + brightness)")
    if args.augment:
        logger.info("Test Time Augmentation ENABLED (slower but catches occluded people)")

    pipeline = SurveillancePipeline(
        zones_config=args.zones,
        output_dir=args.output,
        model_size=args.model,
        confidence=args.confidence,
        device=args.device,
        target_fps=args.target_fps,
        grace_frames=args.grace_frames,
        write_video=not args.no_video,
        enhance=args.enhance,
        augment=args.augment,
    )

    results = pipeline.process_multiple(args.video)

    # Print final summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    for r in results:
        print(f"\nVideo: {r.get('video')}")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  Annotated video : {r.get('annotated_video', 'N/A')}")
        print(f"  Event log (JSON): {r.get('event_log_json')}")
        print(f"  Event log (CSV) : {r.get('event_log_csv')}")
        stats = r.get("event_stats", {})
        print(f"  Events: intrusions={stats.get('intrusions',0)} "
              f"loiterings={stats.get('loiterings',0)} "
              f"deduped={stats.get('deduplicated',0)}")
        pstats = r.get("pipeline_stats", {})
        print(f"  Pipeline FPS: {pstats.get('pipeline_fps')} | "
              f"Avg inference: {pstats.get('avg_inference_ms')}ms")


if __name__ == "__main__":
    main()
