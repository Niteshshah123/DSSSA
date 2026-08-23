"""
Builds a synthetic event scene with a moving, spatially-correlated "object"
(events clustered in local space+time along a path) plus sparse uniform
background noise -- unlike the pure-uniform-noise `synthetic_dataset` used
for M1-M3 plumbing tests, this is what's needed to meaningfully validate
that BAF/isolated-event-removal preserve real signal while removing noise.
"""
import numpy as np


def build_correlated_scene(height=240, width=304, duration_us=200000, seed=2):
    rng = np.random.default_rng(seed)

    n_steps = 400
    path_t = np.linspace(0, duration_us, n_steps)
    path_x = width * 0.15 + (path_t / duration_us) * (width * 0.5)
    path_y = height * 0.4 + (height * 0.15) * np.sin(path_t / duration_us * 3.14159)

    obj_t, obj_x, obj_y = [], [], []
    for pt, px, py in zip(path_t, path_x, path_y):
        n_local = rng.integers(3, 8)
        obj_t.extend(pt + rng.uniform(-50, 50, size=n_local))
        obj_x.extend(px + rng.integers(-3, 4, size=n_local))
        obj_y.extend(py + rng.integers(-3, 4, size=n_local))
    obj_t = np.array(obj_t)
    obj_x = np.clip(np.array(obj_x), 0, width - 1)
    obj_y = np.clip(np.array(obj_y), 0, height - 1)
    n_obj = len(obj_t)

    n_noise = 300
    noise_t = rng.uniform(0, duration_us, size=n_noise)
    noise_x = rng.integers(0, width, size=n_noise)
    noise_y = rng.integers(0, height, size=n_noise)

    is_object = np.concatenate([np.ones(n_obj, dtype=bool), np.zeros(n_noise, dtype=bool)])
    t = np.concatenate([obj_t, noise_t])
    x = np.concatenate([obj_x, noise_x]).astype(np.int64)
    y = np.concatenate([obj_y, noise_y]).astype(np.int64)
    p = rng.integers(0, 2, size=len(t)).astype(np.int64)

    order = np.argsort(t)
    t, x, y, p, is_object = t[order].astype(np.int64), x[order], y[order], p[order], is_object[order]
    return t, x, y, p, is_object
