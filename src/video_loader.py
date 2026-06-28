"""
src/video_loader.py — Multi-Stream Video Loader
=================================================
Handles loading, validation, and frame extraction from
four directional traffic camera video files.

Supports:
  - Local .mp4 files
  - RTSP live streams (just pass the rtsp:// URL as path)
  - Frame skipping for performance
  - Thread-safe multi-direction loading
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Generator, Tuple, List
from dataclasses import dataclass, field
from loguru import logger

from config import Config


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoInfo:
    """Metadata for a single video stream."""
    direction: str
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    is_valid: bool = False

    def __str__(self) -> str:
        return (
            f"VideoInfo({self.direction}: {self.width}x{self.height} "
            f"@ {self.fps:.1f}fps, {self.total_frames} frames, "
            f"{self.duration_seconds:.1f}s)"
        )


@dataclass
class FrameData:
    """A single extracted video frame with metadata."""
    direction: str
    frame: np.ndarray
    frame_number: int
    timestamp: float          # seconds from video start
    wall_time: float          # time.time() when captured
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE STREAM READER
# ═══════════════════════════════════════════════════════════════════════════════

class VideoStream:
    """
    Reads frames from a single video source (file or RTSP).

    Usage:
        stream = VideoStream("north", "data/input/north.mp4")
        stream.open()
        for frame_data in stream.frames(skip=2):
            process(frame_data)
        stream.close()
    """

    def __init__(self, direction: str, path: str, config: Config = None):
        """
        Args:
            direction: Camera direction label (north/south/east/west).
            path:      Path to video file or RTSP URL.
            config:    Project config object.
        """
        self.direction = direction
        self.path = str(path)
        self.config = config or Config()
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._frame_count: int = 0
        self._lock = threading.Lock()

    # ── Open / Close ──────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the video capture.

        Returns:
            True if successfully opened, False otherwise.
        """
        try:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                logger.error(f"[{self.direction}] Cannot open video: {self.path}")
                return False

            self._info = self._read_metadata()
            logger.info(f"[{self.direction}] Opened → {self._info}")
            return True

        except Exception as exc:
            logger.error(f"[{self.direction}] VideoStream.open() failed: {exc}")
            return False

    def close(self) -> None:
        """Release the video capture handle."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.debug(f"[{self.direction}] Stream closed after {self._frame_count} frames.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _read_metadata(self) -> VideoInfo:
        """Read video properties from the capture object."""
        if self._cap is None:
            return VideoInfo(direction=self.direction, path=self.path)

        w     = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur   = total / fps if fps > 0 else 0.0

        return VideoInfo(
            direction=self.direction,
            path=self.path,
            width=w,
            height=h,
            fps=fps,
            total_frames=total,
            duration_seconds=dur,
            is_valid=True,
        )

    @property
    def info(self) -> Optional[VideoInfo]:
        """Return cached VideoInfo (available after open())."""
        return self._info

    # ── Frame Iteration ───────────────────────────────────────────────────────

    def frames(
        self,
        skip: int = 1,
        resize: Optional[Tuple[int, int]] = None,
        loop: bool = False,
    ) -> Generator[FrameData, None, None]:
        """
        Generator that yields FrameData objects.

        Args:
            skip:   Yield every Nth frame (1 = every frame, 2 = every other frame).
            resize: Optional (width, height) to resize each frame.
            loop:   If True, restart from beginning when video ends (for demos).

        Yields:
            FrameData for each selected frame.
        """
        if self._cap is None:
            logger.error(f"[{self.direction}] Call open() before iterating frames.")
            return

        local_frame_idx = 0
        fps = self._info.fps if self._info else 25.0

        while True:
            with self._lock:
                ret, raw_frame = self._cap.read()

            if not ret:
                if loop:
                    logger.debug(f"[{self.direction}] Looping video.")
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    local_frame_idx = 0
                    continue
                else:
                    logger.info(f"[{self.direction}] End of video at frame {self._frame_count}.")
                    break

            local_frame_idx += 1
            self._frame_count += 1

            # Frame skipping
            if (local_frame_idx % skip) != 0:
                continue

            # Optional resize
            if resize:
                raw_frame = cv2.resize(raw_frame, resize, interpolation=cv2.INTER_LINEAR)

            ts = local_frame_idx / fps

            yield FrameData(
                direction=self.direction,
                frame=raw_frame,
                frame_number=self._frame_count,
                timestamp=ts,
                wall_time=time.time(),
                width=raw_frame.shape[1],
                height=raw_frame.shape[0],
            )

    def read_frame(self) -> Optional[FrameData]:
        """
        Read a single frame (non-generator use).

        Returns:
            FrameData or None if video ended or error.
        """
        if self._cap is None:
            return None

        with self._lock:
            ret, raw_frame = self._cap.read()

        if not ret:
            return None

        self._frame_count += 1
        fps = self._info.fps if self._info else 25.0

        return FrameData(
            direction=self.direction,
            frame=raw_frame,
            frame_number=self._frame_count,
            timestamp=self._frame_count / fps,
            wall_time=time.time(),
            width=raw_frame.shape[1],
            height=raw_frame.shape[0],
        )

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frames_read(self) -> int:
        return self._frame_count


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STREAM LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiStreamLoader:
    """
    Manages loading from all four directional video streams.

    Usage:
        loader = MultiStreamLoader()
        loader.open_all()
        for direction, frame_data in loader.iter_frames_round_robin():
            process(frame_data)
        loader.close_all()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._streams: Dict[str, VideoStream] = {}
        self._opened: List[str] = []

    def open_all(self, directions: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Open all (or specified) direction streams.

        Args:
            directions: Subset of directions to open. Defaults to all four.

        Returns:
            Dict mapping direction → success bool.
        """
        directions = directions or self.config.video.directions
        results = {}

        for direction in directions:
            path = self.config.paths.video_paths.get(direction)
            if path is None:
                logger.warning(f"No video path configured for direction: {direction}")
                results[direction] = False
                continue

            if not Path(str(path)).exists():
                logger.warning(
                    f"[{direction}] Video file not found: {path}. "
                    f"Run: python main.py --generate-test-videos"
                )
                results[direction] = False
                continue

            stream = VideoStream(direction, str(path), self.config)
            if stream.open():
                self._streams[direction] = stream
                self._opened.append(direction)
                results[direction] = True
            else:
                results[direction] = False

        opened = sum(results.values())
        logger.info(f"MultiStreamLoader: {opened}/{len(directions)} streams opened.")
        return results

    def close_all(self) -> None:
        """Release all open video captures."""
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        self._opened.clear()
        logger.info("All streams closed.")

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, *args):
        self.close_all()

    # ── Frame Iterators ───────────────────────────────────────────────────────

    def iter_frames_round_robin(
        self,
        skip: int = 1,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Generator[Tuple[str, FrameData], None, None]:
        """
        Yield frames from each direction in round-robin order.
        Stops when ALL streams are exhausted.

        Yields:
            (direction, FrameData) tuples.
        """
        generators = {
            d: s.frames(skip=skip, resize=resize)
            for d, s in self._streams.items()
        }
        active = set(generators.keys())

        while active:
            for direction in list(active):
                try:
                    frame_data = next(generators[direction])
                    yield direction, frame_data
                except StopIteration:
                    logger.info(f"[{direction}] Stream exhausted.")
                    active.discard(direction)

    def read_one_frame_per_direction(
        self,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Optional[FrameData]]:
        """
        Read exactly one frame from each stream simultaneously.

        Returns:
            Dict of direction → FrameData (or None if stream ended).
        """
        results: Dict[str, Optional[FrameData]] = {}
        resize_wh = resize or (self.config.video.frame_width, self.config.video.frame_height)

        for direction, stream in self._streams.items():
            fd = stream.read_frame()
            if fd is not None and resize:
                fd.frame = cv2.resize(fd.frame, resize_wh, interpolation=cv2.INTER_LINEAR)
                fd.width, fd.height = resize_wh
            results[direction] = fd

        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, VideoInfo]:
        """Return VideoInfo for all open streams."""
        return {d: s.info for d, s in self._streams.items() if s.info}

    def validate_videos(self, directions: Optional[List[str]] = None) -> bool:
        """
        Check that all video files exist and are readable.

        Returns:
            True if all required videos are valid.
        """
        directions = directions or self.config.video.directions
        all_valid = True

        for direction in directions:
            path = self.config.paths.video_paths.get(direction)
            if path is None or not Path(str(path)).exists():
                logger.error(f"❌ Missing video: {path}")
                all_valid = False
                continue

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.error(f"❌ Cannot open: {path}")
                all_valid = False
            else:
                ret, _ = cap.read()
                if not ret:
                    logger.error(f"❌ Cannot read frames from: {path}")
                    all_valid = False
                else:
                    logger.success(f"✅ Valid: {path}")
            cap.release()

        return all_valid

    @property
    def open_directions(self) -> List[str]:
        """List of currently open stream directions."""
        return list(self._streams.keys())

    @property
    def stream_count(self) -> int:
        return len(self._streams)
