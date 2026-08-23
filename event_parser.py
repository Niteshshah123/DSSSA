"""
================================================================================
EventParser Module
Event-Aware Dynamic Patch Selection (EDPS) Project - Module M1
================================================================================

This module is the ONLY place in the project that knows how to read raw event-
camera recordings and their annotations off disk. Every later module (voxel
grid generation, noise reduction, normalization, patching, EDPS, etc.) consumes
plain numpy arrays returned by this module and never touches file formats
directly. This separation is deliberate: it is what lets the pipeline later
support a different sensor/dataset (Gen4, DSEC, N-Caltech101, ...) by writing
one new reader class, without touching anything downstream.


--------------------------------------------------------------------------------
1. BINARY FILE FORMAT (Prophesee Gen1 Automotive Detection Dataset, "*_td.dat")
--------------------------------------------------------------------------------

A "*_td.dat" file has exactly three sections, back to back, with NO padding
between them:

    [ASCII HEADER]  [2-BYTE BINARY SUB-HEADER]  [EVENT STREAM]

(a) ASCII HEADER
    A variable number of newline-terminated text lines, each starting with
    the '%' character. Example, taken directly from this dataset:

        % Data file containing TD/APS events.
        % Version 2
        % Date 2019-11-27 09:41:27
        % Height 240
        % Width 304

    The 'Height' and 'Width' lines give the sensor resolution and MUST be
    read from the file rather than hard-coded, since different recordings
    or dataset variants could in principle differ. The header ends at the
    first line that does NOT start with '%'.

(b) 2-BYTE BINARY SUB-HEADER  <-- the part that caused M0's decoding failure
    Immediately after the last '%' line (no separating newline consumed
    beyond what already terminated that line), there are exactly two raw
    bytes:

        byte[0] = event_type  (uint8) -- always 0 for this dataset (CD/TD events)
        byte[1] = event_size  (uint8) -- size in bytes of each event record
                                          that follows (always 8 for this format)

    WHY THIS EXISTS: Prophesee's dat format is a thin, generic container.
    The 2-byte sub-header lets a reader confirm the event record layout
    (via event_size) and event type WITHOUT hard-coding assumptions, so the
    same container format can, in principle, carry different event record
    types. Our M0 script computed the byte offset for the event stream by
    summing the header line lengths and stopped there -- it did not know
    about, and therefore did not skip, these 2 bytes. That constant 2-byte
    offset error corrupted every single event read from the file (garbage
    y-values, non-monotonic timestamps, polarity collapsing to a single
    value), which is exactly the failure mode observed and diagnosed in
    M0 / M0.5.

(c) EVENT STREAM
    A flat sequence of fixed-size event records, each occupying exactly
    `event_size` bytes (8 bytes for this dataset), with NO delimiters
    between records. Each record is:

        bytes [0:4]  timestamp   uint32, little-endian, UNIT = microseconds
                                  Measured from an arbitrary reference point
                                  at the start of the ORIGINAL long recording
                                  session, NOT from the start of this cut
                                  file -- so absolute values can be large,
                                  but within a single file they are always
                                  monotonically non-decreasing.

        bytes [4:8]  packed data  uint32, little-endian, containing three
                                  bit-packed fields:

                                      bits  0-13  (14 bits) -> x  (0 .. 16383 range,
                                                                    but constrained to
                                                                    [0, sensor_width-1]
                                                                    by the sensor)
                                      bits 14-27  (14 bits) -> y  (constrained to
                                                                    [0, sensor_height-1])
                                      bit     28  (1 bit)   -> polarity
                                                                    0 = OFF event
                                                                        (brightness
                                                                        DECREASE)
                                                                    1 = ON event
                                                                        (brightness
                                                                        INCREASE)
                                      bits 29-31            -> unused / reserved,
                                                                    always 0 in this
                                                                    dataset

    Decoding: given the packed uint32 value `d`,

        x = d & 0x3FFF
        y = (d >> 14) & 0x3FFF
        p = (d >> 28) & 0x1

This encoding is confirmed against Prophesee's own published dataset
documentation, and independently cross-checked visually in M0.5 (decoded
event clusters spatially align with ground-truth bounding boxes).


--------------------------------------------------------------------------------
2. ANNOTATION FILE FORMAT ("*_bbox.npy")
--------------------------------------------------------------------------------

A structured numpy array (loadable directly with `np.load`), one row per
labeled bounding box, with fields (confirmed against real dataset files
in M0):

    ts          uint64   timestamp in microseconds, SAME clock/reference as
                          the event stream's timestamp field, so `ts` can be
                          directly compared against decoded event `t` values.
    x, y        float32  top-left corner of the box, in pixel coordinates
    w, h        float32  width / height of the box, in pixels
    class_id    uint8    0 = car, 1 = pedestrian (per Prophesee Gen1 spec)
    confidence  float32  always 1.0 for ground-truth (this field exists so
                          the same schema can also hold model PREDICTIONS,
                          where confidence is meaningful)
    track_id    uint32   identifies the same physical object across frames
                          (NOT globally unique -- do not rely on it for
                          cross-recording identity)


--------------------------------------------------------------------------------
3. DESIGN FOR FUTURE DATASET COMPATIBILITY
--------------------------------------------------------------------------------

`BaseEventReader` is an abstract interface with exactly three responsibilities:
parse a header, read events, and report sensor resolution. `PropheseeGen1Reader`
is the ONLY class in this file that knows about ".dat" files, the 2-byte
sub-header, or the 14/14/1 bit-packing scheme above.

`EventParser` never talks to files directly -- it delegates all of that to
whichever reader it was given (`PropheseeGen1Reader` by default). To support
a new dataset (e.g. Gen4, which uses a different EVT2/EVT3 RAW encoding, or
DSEC, which ships HDF5 files), implement a new `BaseEventReader` subclass and
pass it to `EventParser(reader=YourNewReader())`. No other project code
(voxel grid, EDPS, Transformer, training loop, etc.) needs to change.
================================================================================
"""

from __future__ import annotations

import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np


class ParserValidationError(Exception):
    """
    Raised whenever decoded event or annotation data fails a sanity check.
    Carries a human-readable, specific explanation rather than a bare
    assertion failure, since this is meant to surface real format bugs
    (like the M0 2-byte offset bug) immediately and unambiguously.
    """
    pass


@dataclass
class SensorResolution:
    width: int
    height: int


@dataclass
class EventStatistics:
    """Container for the performance/dataset statistics required by the SRS."""
    total_events: int
    duration_us: int
    duration_s: float
    event_rate_hz: float
    avg_events_per_second: float
    sensor_width: int
    sensor_height: int
    num_annotations: Optional[int] = None
    polarity_counts: Optional[Dict[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# ABSTRACT READER INTERFACE (for future dataset compatibility)
# ==============================================================================
class BaseEventReader(ABC):
    """
    Abstract interface every dataset-specific reader must implement.
    See module docstring, Section 3, for the rationale.
    """

    @abstractmethod
    def parse_header(self, path: str) -> Dict[str, Any]:
        """Parse whatever header/metadata precedes the event stream."""
        raise NotImplementedError

    @abstractmethod
    def read_events(
        self,
        path: str,
        header_info: Dict[str, Any],
        max_events: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, x, y, p) numpy arrays for the events in `path`."""
        raise NotImplementedError

    @abstractmethod
    def get_resolution(self, header_info: Dict[str, Any]) -> SensorResolution:
        """Return the sensor resolution described by `header_info`."""
        raise NotImplementedError

    @abstractmethod
    def get_time_range(self, path: str, header_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Return (t_min, t_max, n_events) WITHOUT decoding every event."""
        raise NotImplementedError


# ==============================================================================
# PROPHESEE GEN1 READER (the only class that knows about the .dat binary layout)
# ==============================================================================
class PropheseeGen1Reader(BaseEventReader):
    """
    Reader for Prophesee Gen1 Automotive Detection Dataset "*_td.dat" files.
    See the module docstring for the full binary format documentation.
    """

    EVENT_RECORD_DTYPE = np.dtype([("t", "<u4"), ("_", "<u4")])
    X_MASK = 0x3FFF
    Y_SHIFT = 14
    Y_MASK = 0x3FFF
    P_SHIFT = 28
    P_MASK = 0x1
    EXPECTED_EVENT_SIZE = 8

    def parse_header(self, path: str) -> Dict[str, Any]:
        header_lines = []
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    raise ParserValidationError(
                        f"{path}: reached end-of-file while still reading the "
                        f"ASCII header -- file may be empty, truncated, or not "
                        f"in the expected Prophesee dat format."
                    )
                if line.startswith(b"%"):
                    header_lines.append(line.decode("latin-1").rstrip())
                else:
                    f.seek(pos)
                    break

            two_bytes = f.read(2)
            if len(two_bytes) < 2:
                raise ParserValidationError(
                    f"{path}: file ended before the required 2-byte "
                    f"(event_type, event_size) sub-header could be read. "
                    f"This is the exact bug diagnosed in M0 -- do not skip this."
                )
            event_type, event_size = struct.unpack("<BB", two_bytes)
            data_start_offset = f.tell()

        height, width = None, None
        for line in header_lines:
            tokens = line.split()
            if "Height" in line and tokens:
                height = int(tokens[-1])
            if "Width" in line and tokens:
                width = int(tokens[-1])

        if height is None or width is None:
            raise ParserValidationError(
                f"{path}: could not find both 'Height' and 'Width' fields in "
                f"the ASCII header. Header lines found: {header_lines}"
            )

        if event_size != self.EXPECTED_EVENT_SIZE:
            raise ParserValidationError(
                f"{path}: sub-header declares event_size={event_size} bytes, "
                f"but this reader only supports event_size="
                f"{self.EXPECTED_EVENT_SIZE}. This file may use a different "
                f"Prophesee format version (e.g. EVT2/EVT3 RAW) that requires "
                f"a different BaseEventReader implementation."
            )

        return {
            "header_lines": header_lines,
            "event_type": event_type,
            "event_size": event_size,
            "data_start_offset": data_start_offset,
            "height": height,
            "width": width,
            "file_size": os.path.getsize(path),
        }

    def _decode_raw_events(self, raw):
        """Decode packed Gen1 records into (t, x, y, p)."""
        if raw.size == 0:
            raise ParserValidationError("No events were decoded for the requested range.")

        t = raw["t"].astype(np.int64)
        x = (raw["_"] & self.X_MASK).astype(np.int32)
        y = ((raw["_"] >> self.Y_SHIFT) & self.Y_MASK).astype(np.int32)
        p = ((raw["_"] >> self.P_SHIFT) & self.P_MASK).astype(np.int8)
        return t, x, y, p

    def read_events(self, path, header_info, max_events=None):
        with open(path, "rb") as f:
            f.seek(header_info["data_start_offset"])
            raw = np.fromfile(
                f,
                dtype=self.EVENT_RECORD_DTYPE,
                count=max_events if max_events else -1,
            )

        return self._decode_raw_events(raw)

    def read_events_window(self, path, header_info, t_start, t_end):
        """
        Read only the event records whose timestamps satisfy
        t_start <= t < t_end.

        Gen1 event timestamps are chronologically ordered, so a memory-mapped
        timestamp field lets us locate the two boundaries with binary search
        without decoding the entire recording. Only the requested records are
        then copied and decoded.

        This is critical for M10: a 50 ms training window must not require
        loading an entire multi-second / multi-million-event recording.
        """
        if t_end <= t_start:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int8),
            )

        data_bytes = header_info["file_size"] - header_info["data_start_offset"]
        n_events = data_bytes // header_info["event_size"]
        if n_events <= 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int8),
            )

        mm = np.memmap(
            path,
            dtype=self.EVENT_RECORD_DTYPE,
            mode="r",
            offset=header_info["data_start_offset"],
            shape=(n_events,),
        )
        try:
            timestamps = mm["t"]
            start_idx = int(np.searchsorted(timestamps, np.uint64(t_start), side="left"))
            end_idx = int(np.searchsorted(timestamps, np.uint64(t_end), side="left"))

            if end_idx <= start_idx:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int32),
                    np.empty(0, dtype=np.int8),
                )

            raw = np.asarray(mm[start_idx:end_idx]).copy()
        finally:
            del mm

        return self._decode_raw_events(raw)

    def get_resolution(self, header_info: Dict[str, Any]) -> SensorResolution:
        return SensorResolution(width=header_info["width"], height=header_info["height"])

    def get_time_range(self, path: str, header_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """
        Returns (t_min, t_max, n_events) using only two targeted seeks (first
        and last event record) plus file-size arithmetic for the count --
        avoids decoding the whole file just to learn its duration/rate.
        """
        data_bytes = header_info["file_size"] - header_info["data_start_offset"]
        n_events = data_bytes // header_info["event_size"]
        if n_events == 0:
            return 0, 0, 0
        with open(path, "rb") as f:
            f.seek(header_info["data_start_offset"])
            t_min, _ = struct.unpack("<II", f.read(8))
            f.seek(header_info["file_size"] - header_info["event_size"])
            t_max, _ = struct.unpack("<II", f.read(8))
        return int(t_min), int(t_max), int(n_events)


# ==============================================================================
# UNIFIED PUBLIC INTERFACE
# ==============================================================================
class EventParser:
    """
    Dataset-agnostic public interface used by every other module in the
    pipeline. Internally delegates all format-specific work to a
    `BaseEventReader` implementation (default: `PropheseeGen1Reader`).

    Example
    -------
        parser = EventParser()
        events = parser.load_events("recording_td.dat")
        boxes = parser.load_annotations("recording_bbox.npy")
        stats = parser.get_event_statistics(events, boxes=boxes)
        parser.visualize_events(events, boxes=boxes, out_path="check.png")
    """

    REQUIRED_ANNOTATION_FIELDS = {"x", "y", "w", "h", "class_id"}

    def __init__(self, reader: Optional[BaseEventReader] = None, validate_on_load: bool = True):
        self.reader: BaseEventReader = reader or PropheseeGen1Reader()
        self.validate_on_load = validate_on_load

    # ------------------------------------------------------------------
    def load_events(
        self,
        dat_path: str,
        max_events: Optional[int] = None,
        validate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Load and decode an event recording.

        Returns a dict: {"t": ndarray, "x": ndarray, "y": ndarray,
                          "p": ndarray, "header": dict}
        Raises ParserValidationError if `validate` (or the instance default)
        is True and any sanity check fails.
        """
        validate = self.validate_on_load if validate is None else validate

        header_info = self.reader.parse_header(dat_path)
        t, x, y, p = self.reader.read_events(dat_path, header_info, max_events=max_events)

        if validate:
            self._validate_events(t, x, y, p, header_info, source=dat_path)

        return {"t": t, "x": x, "y": y, "p": p, "header": header_info}

    def load_events_window(
        self,
        dat_path: str,
        t_start: int,
        t_end: int,
        validate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Load only one temporal window from a recording.

        Falls back to full-file loading only for reader implementations that
        do not expose read_events_window().
        """
        validate = self.validate_on_load if validate is None else validate
        header_info = self.reader.parse_header(dat_path)

        window_reader = getattr(self.reader, "read_events_window", None)
        if window_reader is not None:
            t, x, y, p = window_reader(dat_path, header_info, t_start, t_end)
        else:
            events = self.reader.read_events(dat_path, header_info)
            mask = (events[0] >= t_start) & (events[0] < t_end)
            t, x, y, p = (a[mask] for a in events)

        if validate and len(t) > 0:
            self._validate_events(t, x, y, p, header_info, source=dat_path)

        return {"t": t, "x": x, "y": y, "p": p, "header": header_info}

    def _validate_events(self, t, x, y, p, header_info, source: str = "<unknown>"):
        if len(t) == 0:
            raise ParserValidationError(f"{source}: decoded zero events.")

        if not np.all(np.diff(t) >= 0):
            n_bad = int(np.sum(np.diff(t) < 0))
            raise ParserValidationError(
                f"{source}: timestamps are not monotonically increasing "
                f"({n_bad} violation(s) out of {len(t) - 1} consecutive pairs). "
                f"This is the signature of a byte-alignment bug in the reader."
            )

        w, h = header_info["width"], header_info["height"]
        if x.min() < 0 or x.max() > w - 1:
            raise ParserValidationError(
                f"{source}: x out of bounds. Observed [{x.min()}, {x.max()}], "
                f"expected [0, {w - 1}] for declared sensor width {w}."
            )
        if y.min() < 0 or y.max() > h - 1:
            raise ParserValidationError(
                f"{source}: y out of bounds. Observed [{y.min()}, {y.max()}], "
                f"expected [0, {h - 1}] for declared sensor height {h}."
            )

        unique_p = set(np.unique(p).tolist())
        if not unique_p.issubset({0, 1}):
            raise ParserValidationError(
                f"{source}: polarity values must be a subset of {{0, 1}}, "
                f"found: {unique_p}."
            )

    # ------------------------------------------------------------------
    def load_annotations(self, bbox_path: str) -> np.ndarray:
        """Load and schema-check a bounding-box annotation file."""
        if not os.path.exists(bbox_path):
            raise ParserValidationError(f"Annotation file not found: {bbox_path}")

        boxes = np.load(bbox_path, allow_pickle=True)

        if boxes.dtype.names is None:
            raise ParserValidationError(
                f"{bbox_path}: expected a structured numpy array with named "
                f"fields, but got dtype={boxes.dtype}."
            )

        missing = self.REQUIRED_ANNOTATION_FIELDS - set(boxes.dtype.names)
        if missing:
            raise ParserValidationError(
                f"{bbox_path}: annotation schema is missing required "
                f"field(s) {missing}. Fields found: {boxes.dtype.names}."
            )

        return boxes

    # ------------------------------------------------------------------
    def get_sensor_resolution(
        self,
        dat_path: Optional[str] = None,
        header_info: Optional[Dict[str, Any]] = None,
    ) -> SensorResolution:
        """Get sensor resolution, either from a raw file path or a pre-parsed header."""
        if header_info is None:
            if dat_path is None:
                raise ValueError("Provide either dat_path or header_info.")
            header_info = self.reader.parse_header(dat_path)
        return self.reader.get_resolution(header_info)

    # ------------------------------------------------------------------
    def get_event_statistics(
        self,
        events: Dict[str, Any],
        boxes: Optional[np.ndarray] = None,
    ) -> EventStatistics:
        """
        Compute the dataset/performance statistics required for reporting:
        total events, duration, event rate, avg events/sec, sensor
        resolution, and (optionally) annotation count.
        """
        t = events["t"]
        header = events["header"]

        duration_us = int(t[-1] - t[0]) if len(t) > 1 else 0
        duration_s = duration_us / 1e6 if duration_us > 0 else 0.0
        total_events = int(len(t))
        event_rate_hz = (total_events / duration_s) if duration_s > 0 else 0.0

        unique_p, counts_p = np.unique(events["p"], return_counts=True)
        polarity_counts = {int(k): int(v) for k, v in zip(unique_p, counts_p)}

        return EventStatistics(
            total_events=total_events,
            duration_us=duration_us,
            duration_s=duration_s,
            event_rate_hz=event_rate_hz,
            avg_events_per_second=event_rate_hz,
            sensor_width=header["width"],
            sensor_height=header["height"],
            num_annotations=int(len(boxes)) if boxes is not None else None,
            polarity_counts=polarity_counts,
        )

    # ------------------------------------------------------------------
    def visualize_events(
        self,
        events: Dict[str, Any],
        boxes: Optional[np.ndarray] = None,
        window_start_us: Optional[int] = None,
        window_us: int = 30000,
        out_path: str = "event_visualization.png",
        show: bool = False,
    ) -> str:
        """
        Render a single accumulated event frame (polarity color-coded) over
        a time window, optionally overlaid with matching ground-truth boxes.
        Saves a PNG to `out_path` and returns that path.
        """
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        t, x, y, p = events["t"], events["x"], events["y"], events["p"]
        header = events["header"]
        w, h = header["width"], header["height"]

        if window_start_us is None:
            window_start_us = int(t[0])

        mask = (t >= window_start_us) & (t < window_start_us + window_us)
        xs, ys, ps = x[mask], y[mask], p[mask]

        frame = np.full((h, w, 3), 255, dtype=np.uint8)
        frame[ys[ps == 1], xs[ps == 1]] = [255, 0, 0]
        frame[ys[ps == 0], xs[ps == 0]] = [0, 0, 255]

        fig, ax = plt.subplots(figsize=(8, max(8 * h / w, 3)))
        ax.imshow(frame)

        if boxes is not None and len(boxes) > 0:
            ts_field = "ts" if "ts" in boxes.dtype.names else "t"
            box_mask = (boxes[ts_field] >= window_start_us) & (
                boxes[ts_field] < window_start_us + window_us + 200000
            )
            for b in boxes[box_mask]:
                rect = patches.Rectangle(
                    (b["x"], b["y"]), b["w"], b["h"],
                    linewidth=1.5, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(b["x"], max(b["y"] - 3, 0), f"cls={b['class_id']}",
                         color="lime", fontsize=8)

        ax.set_title(f"Event frame [{window_start_us}-{window_start_us + window_us}us]")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return out_path
