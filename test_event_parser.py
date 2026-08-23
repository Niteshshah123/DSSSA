"""
================================================================================
Unit Tests for EventParser (M1)
================================================================================
Run standalone against a real dataset file pair:

    python test_event_parser.py --dat path/to/xxx_td.dat --bbox path/to/xxx_bbox.npy

Or import and call run_all_tests(dat_path, bbox_path) from another script
(used by generate_m1_report.py to fold results into the research log).

Each test fails with a specific, human-readable message rather than a bare
assertion, per the SRS requirement that the parser "fail with meaningful
error messages if any validation fails."
================================================================================
"""

import os
import argparse
import numpy as np

from event_parser import EventParser, ParserValidationError


class TestResult:
    def __init__(self):
        self.results = []

    def record(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")

    def summary(self) -> bool:
        n_pass = sum(1 for _, p, _ in self.results if p)
        n_total = len(self.results)
        print(f"\n{n_pass}/{n_total} tests passed.")
        return n_pass == n_total


def run_all_tests(dat_path: str, bbox_path: str, cleanup_visualization: bool = True):
    """
    Runs the full required test battery:
      1. header parsing
      2. event count
      3. timestamp ordering
      4. x/y bounds
      5. polarity values
      6. annotation loading
      7. visualization
      8. full validation pipeline (load_events with validate=True)
    Returns (all_passed: bool, results: list[(name, passed, detail)]).
    """
    print(f"Running EventParser unit tests against:\n  dat:  {dat_path}\n  bbox: {bbox_path}\n")

    parser = EventParser()
    tr = TestResult()
    header_info = None
    events = None
    boxes = None

    # ---- 1. Header parsing ----
    try:
        header_info = parser.reader.parse_header(dat_path)
        ok = (
            header_info["width"] > 0
            and header_info["height"] > 0
            and header_info["event_size"] == 8
        )
        tr.record(
            "header_parsing", ok,
            f"width={header_info['width']}, height={header_info['height']}, "
            f"event_size={header_info['event_size']}, "
            f"data_start_offset={header_info['data_start_offset']}",
        )
    except ParserValidationError as e:
        tr.record("header_parsing", False, str(e))
    except Exception as e:
        tr.record("header_parsing", False, f"Unexpected error: {e}")

    # ---- 2. Event count (loaded WITHOUT validation, to isolate this check) ----
    try:
        events = parser.load_events(dat_path, validate=False)
        ok = len(events["t"]) > 0
        tr.record("event_count", ok, f"{len(events['t'])} events decoded")
    except Exception as e:
        tr.record("event_count", False, str(e))
        events = None

    # ---- 3. Timestamp ordering ----
    if events is not None:
        try:
            n_bad = int(np.sum(np.diff(events["t"]) < 0))
            ok = n_bad == 0
            tr.record("timestamp_ordering", ok,
                       "monotonic" if ok else f"{n_bad} decreasing-timestamp violations")
        except Exception as e:
            tr.record("timestamp_ordering", False, str(e))
    else:
        tr.record("timestamp_ordering", False, "skipped: no events loaded")

    # ---- 4. x/y bounds ----
    if events is not None and header_info is not None:
        try:
            w, h = header_info["width"], header_info["height"]
            x, y = events["x"], events["y"]
            ok = x.min() >= 0 and x.max() <= w - 1 and y.min() >= 0 and y.max() <= h - 1
            tr.record(
                "xy_bounds", ok,
                f"x:[{x.min()},{x.max()}] y:[{y.min()},{y.max()}] vs. sensor w={w},h={h}",
            )
        except Exception as e:
            tr.record("xy_bounds", False, str(e))
    else:
        tr.record("xy_bounds", False, "skipped: no events/header available")

    # ---- 5. Polarity values ----
    if events is not None:
        try:
            unique_p = set(np.unique(events["p"]).tolist())
            ok = unique_p.issubset({0, 1}) and len(unique_p) > 0
            tr.record("polarity_values", ok, f"observed values: {unique_p}")
        except Exception as e:
            tr.record("polarity_values", False, str(e))
    else:
        tr.record("polarity_values", False, "skipped: no events loaded")

    # ---- 6. Annotation loading ----
    try:
        boxes = parser.load_annotations(bbox_path)
        ok = len(boxes) > 0
        tr.record("annotation_loading", ok, f"{len(boxes)} boxes loaded, "
                                             f"fields={boxes.dtype.names}")
    except ParserValidationError as e:
        tr.record("annotation_loading", False, str(e))
    except Exception as e:
        tr.record("annotation_loading", False, f"Unexpected error: {e}")

    # ---- 7. Visualization ----
    try:
        out_path = "test_visualization.png"
        result_path = parser.visualize_events(events, boxes=boxes, out_path=out_path)
        ok = os.path.exists(result_path) and os.path.getsize(result_path) > 0
        tr.record("visualization", ok, f"saved to {result_path} "
                                        f"({os.path.getsize(result_path) if ok else 0} bytes)")
        if cleanup_visualization and ok:
            os.remove(result_path)
    except Exception as e:
        tr.record("visualization", False, str(e))

    # ---- 8. Full validation pipeline (should raise nothing on good data) ----
    try:
        parser.load_events(dat_path, validate=True)
        tr.record("full_validation_pipeline", True, "load_events(validate=True) raised no error")
    except ParserValidationError as e:
        tr.record("full_validation_pipeline", False, str(e))
    except Exception as e:
        tr.record("full_validation_pipeline", False, f"Unexpected error: {e}")

    all_passed = tr.summary()
    return all_passed, tr.results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EventParser unit tests")
    ap.add_argument("--dat", required=True, help="Path to a *_td.dat file")
    ap.add_argument("--bbox", required=True, help="Path to the matching *_bbox.npy file")
    args = ap.parse_args()

    passed, _ = run_all_tests(args.dat, args.bbox)
    raise SystemExit(0 if passed else 1)
