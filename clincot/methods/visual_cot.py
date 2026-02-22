from __future__ import annotations

import re
from typing import Optional

import torch
from PIL import Image

BBOX_PATTERN = r"[-+]?\d*\.?\d+"


def parse_bbox_from_text(text: str) -> Optional[list[float]]:
    """Parse bbox [x1,y1,x2,y2] from raw text.

    Returns normalized bbox if possible, otherwise None.
    """
    nums = re.findall(BBOX_PATTERN, text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in nums[:4]]
    return [x1, y1, x2, y2]


def _to_pixel_coords(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    # If bbox seems normalized, map to pixels.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 = x1 * width
        x2 = x2 * width
        y1 = y1 * height
        y2 = y2 * height

    x1, x2 = int(min(x1, x2)), int(max(x1, x2))
    y1, y2 = int(min(y1, y2)), int(max(y1, y2))

    x1 = max(0, min(x1, width - 1))
    x2 = max(1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(1, min(y2, height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def crop_with_bbox(image: Image.Image, bbox: list[float], min_crop: int = 168) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = _to_pixel_coords(bbox, width=width, height=height)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max((x2 - x1) // 2, (y2 - y1) // 2, min_crop)

    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(width, cx + half)
    bottom = min(height, cy + half)

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)

    return image.crop((left, top, right, bottom))


def build_two_view_tensor(image_processor, image: Image.Image, bbox: list[float]) -> torch.Tensor:
    local = crop_with_bbox(image, bbox)
    px_global = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
    px_local = image_processor(images=local, return_tensors="pt")["pixel_values"][0]
    return torch.stack([px_global, px_local], dim=0)
