from __future__ import annotations

import numpy as np


def render_1d_state(position: float, target: float, *, size: int = 64) -> np.ndarray:
    """Render a tiny 1D manipulation state as a grayscale image."""

    img = np.zeros((size, size), dtype=float)
    rail_y = size // 2
    img[rail_y - 1 : rail_y + 2, 4 : size - 4] = 0.25
    pos_x = int(np.clip(4 + position * (size - 8), 4, size - 5))
    tgt_x = int(np.clip(4 + target * (size - 8), 4, size - 5))
    img[rail_y - 5 : rail_y + 6, max(0, tgt_x - 1) : min(size, tgt_x + 2)] = 0.65
    img[rail_y - 4 : rail_y + 5, max(0, pos_x - 4) : min(size, pos_x + 5)] = 1.0
    return img
