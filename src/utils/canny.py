#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

"""Canny edges without opencv-python.

Matches ``cv2.Canny(image, low, high)`` defaults: 3x3 Sobel, L1 gradient,
replicate borders, and 8-connected hysteresis. Multi-channel input uses the
channel with the largest gradient at each pixel, which is what OpenCV does
when the diffusers ControlNet example passes an RGB image straight through.
"""

import math

import numpy as np
from scipy import ndimage


def canny_edges(image: np.ndarray, low_threshold: float = 100, high_threshold: float = 200) -> np.ndarray:
    """Return a uint8 edge map with values 0 or 255."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3 or image.shape[2] == 0:
        raise ValueError(f"Canny input must be HxW or HxWxC, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Canny input must be uint8, got {image.dtype}")

    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)
    if low_threshold > high_threshold:
        low_threshold, high_threshold = high_threshold, low_threshold
    low = math.floor(low_threshold)
    high = math.floor(high_threshold)

    dx, dy = _sobel_3x3(image.astype(np.int32))
    magnitude = np.abs(dx) + np.abs(dy)
    channel = np.argmax(magnitude, axis=2)
    rows = np.arange(image.shape[0])[:, None]
    cols = np.arange(image.shape[1])[None, :]
    magnitude = magnitude[rows, cols, channel]
    dx = dx[rows, cols, channel]
    dy = dy[rows, cols, channel]

    keep = _non_max_suppression(magnitude, dx, dy)
    candidates = (magnitude > low) & keep
    strong = candidates & (magnitude > high)
    if not np.any(strong):
        return np.zeros(magnitude.shape, dtype=np.uint8)

    labels, _ = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.int8))
    edges = np.isin(labels, np.unique(labels[strong]))
    return edges.astype(np.uint8) * 255


def _sobel_3x3(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV Sobel ksize=3 with BORDER_REPLICATE, per channel."""
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    dy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return dx, dy


def _non_max_suppression(magnitude: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Quantize gradient direction the way OpenCV Canny does (TG22 = 13573)."""
    padded = np.pad(magnitude, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    x = np.abs(dx).astype(np.int64)
    y = np.abs(dy).astype(np.int64) << 15
    tg22x = x * 13573
    horizontal = y < tg22x
    vertical = y > tg22x + (x << 16)
    diagonal = ~(horizontal | vertical)

    signs_differ = (dx.astype(np.int64) ^ dy.astype(np.int64)) < 0
    diag_prev = np.where(signs_differ, up_right, up_left)
    diag_next = np.where(signs_differ, down_left, down_right)

    keep = np.zeros(magnitude.shape, dtype=bool)
    keep[horizontal] = ((center > left) & (center >= right))[horizontal]
    keep[vertical] = ((center > up) & (center >= down))[vertical]
    keep[diagonal] = ((center > diag_prev) & (center > diag_next))[diagonal]
    return keep
