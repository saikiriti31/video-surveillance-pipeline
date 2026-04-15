"""
SurveillancePipeline — main orchestrator.

Ties together: Detector → Tracker → ZoneEngine → EventEngine → Output

Design principles:
  - Clean separation of concerns (each module is independently testable)
  - Memory-efficient: frames are not accumulated; processed one at a time
  - GPU/CPU aware via detector configuration
  - Configurable frame sampling for real-time / near-real-time scenarios
  - Graceful error handling at every stage
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional

import cv2

from .detector import PersonDetector
from .zones import ZoneManager
from .events import EventEngine
from .output import AnnotatedVideoWriter, EventLogger
from .utils import PipelineStats, FrameSampler, get_video_info

logger = logging.getLogger(__name__)


class SurveillancePipeline:
    """
    End-to-end video surveillance pipeline.

    Parameters
    ----------
    zones_config : str
        Path to zones.json file.
    output_dir : str
        Directory for output video and event logs.
    model_size : str
        YOLOv9 model size: 't'|'s'|'m'|'c'|'e'. Default 't' (tiny).
        t=tiny (CPU), s=small, m=medium (GPU recommended), c=compact, e=extended.
    confidence : float
        Detection confidence threshold. Default 0.3.
    device : str, optional
        'cpu'|'cuda'. Auto-detected if None.
    target_fps : float, optional
        Downsample processing to this FPS. None = process all frames.
    loitering_seconds : float
        Override loitering threshold for all zones. None = use per-zone config.
    grace_frames : int
        Occlusion grace period in frames before track state is expired.
    write_video : bool
        Whether to produce annotated output video. Default True.
    """

    def __init__(
        self,
        zones_config: str,
        output_dir: str = "results",
        model_size: str = "t",
        confidence: float = 0.2,
        device: Optional[str] = None,
        target_fps: Optional[float] = None,
        grace_frames: int = 15,
        write_video: bool = True,
        enhance: bool = False,
        augment: bool = False,
    ):
        self.zones_config = zones_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.confidence = confidence
        self.device = device
        self.target_fps = target_fps
        self.grace_frames = grace_frames
        self.write_video = write_video
        self.enhance = enhance
        self.augment = augment

        # Initialize subsystems
        logger.info("Initializing SurveillancePipeline...")
        self.zone_manager = ZoneManager(zones_config)
        self.detector = PersonDetector(
            model_size=model_size,
            confidence=confidence,
            device=device,
            enhance=enhance,
            augment=augment,
        )
        logger.info("Pipeline initialized successfully")

    def process_video(self, video_path: str) -> dict:
        """
        Process a single video file end-to-end.

        Returns a summary dict with paths to output files and stats.
        """
        video_path = str(video_path)
        video_name = Path(video_path).stem

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_path}")

        # Get video metadata
        video_info = get_video_info(video_path)
        logger.info(
            f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps "
            f"| {video_info.total_frames} frames | {video_info.duration_seconds:.1f}s"
        )

        # Per-video subsystems (reset state between videos)
        fps = video_info.fps
        sampler = FrameSampler(fps, target_fps=self.target_fps)
        event_engine = EventEngine(
            zone_manager=self.zone_manager,
            fps=fps,
            grace_frames=self.grace_frames,
        )
        event_logger = EventLogger(
            output_dir=str(self.output_dir),
            video_name=video_name,
        )
        stats = PipelineStats(video_path=video_path)

        # Reset detector tracker state for new video
        self.detector.reset_tracker()

        # Output video setup
        video_writer: Optional[AnnotatedVideoWriter] = None
        if self.write_video:
            out_video_path = str(self.output_dir / f"{video_name}_annotated.mp4")
            video_writer = AnnotatedVideoWriter(
                output_path=out_video_path,
                width=video_info.width,
                height=video_info.height,
                fps=fps,
                zone_manager=self.zone_manager,
            )

        # Main processing loop
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_number = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if not sampler.should_process(frame_number):
                    stats.record_skip()
                    # Still write the frame to keep video timing correct
                    if video_writer:
                        video_writer.write_frame(frame, [], frame_number)
                    frame_number += 1
                    continue

                # --- Detection + Tracking ---
                t_inf_start = time.perf_counter()
                detections = self.detector.detect(frame, frame_number, fps)
                inf_ms = (time.perf_counter() - t_inf_start) * 1000

                # --- Event Detection ---
                events = event_engine.process_frame(detections, frame_number)

                # --- Logging ---
                event_logger.log_events(events)
                stats.record_frame(len(detections), inf_ms)

                if events:
                    logger.debug(f"Frame {frame_number}: {len(events)} events | {len(detections)} tracks")

                # --- Output ---
                if video_writer:
                    video_writer.update_events(events)
                    video_writer.write_frame(frame, detections, frame_number)

                frame_number += 1

                # Progress log every 500 frames
                if frame_number % 500 == 0:
                    pct = frame_number / max(video_info.total_frames, 1) * 100
                    logger.info(f"Progress: {frame_number}/{video_info.total_frames} ({pct:.0f}%)")

        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
        finally:
            cap.release()
            if video_writer:
                video_writer.close()

        # Finalize outputs
        event_summary = event_logger.finalize()
        stats.log_summary()

        result = {
            "video": video_path,
            "output_dir": str(self.output_dir),
            "annotated_video": str(self.output_dir / f"{video_name}_annotated.mp4") if self.write_video else None,
            "event_log_json": event_summary["json_path"],
            "event_log_csv": event_summary["csv_path"],
            "event_stats": event_summary["stats"],
            "pipeline_stats": stats.summary(),
        }

        logger.info(f"Completed: {video_path}")
        logger.info(f"Events: {event_summary['stats']}")

        return result

    def process_multiple(self, video_paths: list) -> list:
        """Process multiple videos sequentially, resetting state between each."""
        results = []
        for vp in video_paths:
            try:
                result = self.process_video(vp)
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to process {vp}: {exc}", exc_info=True)
                results.append({"video": vp, "error": str(exc)})
        return results
