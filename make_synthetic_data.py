"""
Builds a tiny synthetic *_td.dat / *_bbox.npy pair that exactly mimics the
real Prophesee Gen1 format (as confirmed in M0/M0.5), so the EventParser
module can be self-tested here before being handed off to run against the
user's real Colab dataset (which this sandbox has no access to).
"""
import struct
import numpy as np

WIDTH = 304
HEIGHT = 240


def write_synthetic_dat(path, n_events=5000, seed=0):
    rng = np.random.default_rng(seed)

    header = (
        "% Data file containing TD/APS events.\n"
        "% Version 2\n"
        "% Date 2019-11-27 09:41:27\n"
        f"% Height {HEIGHT}\n"
        f"% Width {WIDTH}\n"
    ).encode("latin-1")

    sub_header = struct.pack("<BB", 0, 8)  # event_type=0, event_size=8

    t = np.sort(rng.integers(0, 5_000_000, size=n_events)).astype(np.uint32)
    x = rng.integers(0, WIDTH, size=n_events).astype(np.uint32)
    y = rng.integers(0, HEIGHT, size=n_events).astype(np.uint32)
    p = rng.integers(0, 2, size=n_events).astype(np.uint32)

    packed = (x & 0x3FFF) | ((y & 0x3FFF) << 14) | ((p & 0x1) << 28)

    with open(path, "wb") as f:
        f.write(header)
        f.write(sub_header)
        for ti, di in zip(t, packed):
            f.write(struct.pack("<II", int(ti), int(di)))

    return t, x, y, p


def write_synthetic_bbox(path, t, seed=0):
    rng = np.random.default_rng(seed)
    n_boxes = 20
    dtype = np.dtype([
        ("ts", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
        ("class_id", "u1"), ("confidence", "<f4"), ("track_id", "<u4"),
    ])
    boxes = np.zeros(n_boxes, dtype=dtype)
    boxes["ts"] = np.sort(rng.choice(t, size=n_boxes, replace=False))
    boxes["x"] = rng.uniform(0, WIDTH - 50, size=n_boxes)
    boxes["y"] = rng.uniform(0, HEIGHT - 50, size=n_boxes)
    boxes["w"] = rng.uniform(10, 50, size=n_boxes)
    boxes["h"] = rng.uniform(10, 50, size=n_boxes)
    boxes["class_id"] = rng.integers(0, 2, size=n_boxes)
    boxes["confidence"] = 1.0
    boxes["track_id"] = np.arange(n_boxes)
    np.save(path, boxes)


if __name__ == "__main__":
    t, x, y, p = write_synthetic_dat("synthetic_td.dat")
    write_synthetic_bbox("synthetic_bbox.npy", t)
    print("Synthetic files written: synthetic_td.dat, synthetic_bbox.npy")
    print(f"  events: {len(t)}, t range: [{t.min()}, {t.max()}]")
