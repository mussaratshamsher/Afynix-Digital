"""Utility functions for the attendance system."""

import cv2
import numpy as np
from typing import Tuple


def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int],
               bg_color: Tuple[int, int, int] = (0, 255, 0),
               text_color: Tuple[int, int, int] = (0, 0, 0),
               font_scale: float = 0.6, thickness: int = 2) -> np.ndarray:
    """Draw label with background on frame.

    Args:
        frame: Input frame
        text: Text to display
        position: (x, y) position for top-left of text
        bg_color: Background color (BGR)
        text_color: Text color (BGR)
        font_scale: Font scale
        thickness: Text thickness

    Returns:
        Frame with label
    """
    result = frame.copy()

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    x, y = position

    # Draw background rectangle
    cv2.rectangle(
        result,
        (x, y - text_height - 10),
        (x + text_width + 10, y + 5),
        bg_color,
        -1
    )

    # Draw text
    cv2.putText(
        result,
        text,
        (x + 5, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )

    return result


def add_status_bar(frame: np.ndarray, text: str,
                  position: str = "bottom") -> np.ndarray:
    """Add status bar at top or bottom of frame.

    Args:
        frame: Input frame
        text: Status text
        position: "top" or "bottom"

    Returns:
        Frame with status bar
    """
    result = frame.copy()
    height, width = result.shape[:2]

    bar_height = 30

    if position == "top":
        cv2.rectangle(result, (0, 0), (width, bar_height), (0, 0, 0), -1)
        cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    else:
        cv2.rectangle(result, (0, height - bar_height), (width, height), (0, 0, 0), -1)
        cv2.putText(result, text, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return result


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB.

    Args:
        frame: BGR frame

    Returns:
        RGB frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to BGR.

    Args:
        frame: RGB frame

    Returns:
        BGR frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def resize_frame(frame: np.ndarray, width: int = None, height: int = None,
              scale: float = None) -> np.ndarray:
    """Resize frame.

    Args:
        frame: Input frame
        width: Target width
        height: Target height
        scale: Scale factor (alternative to width/height)

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width is not None and height is not None:
        new_w = width
        new_h = height
    elif width is not None:
        new_w = width
        new_h = int(h * (width / w))
    elif height is not None:
        new_h = height
        new_w = int(w * (height / h))
    else:
        return frame

    return cv2.resize(frame, (new_w, new_h))


def put_timestamp(frame: np.ndarray) -> np.ndarray:
    """Add timestamp to frame.

    Args:
        frame: Input frame

    Returns:
        Frame with timestamp
    """
    from datetime import datetime

    result = frame.copy()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cv2.putText(result, timestamp, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result