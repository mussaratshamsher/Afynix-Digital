"""Camera module for capturing video from webcam."""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Handles webcam video capture."""

    def __init__(self, camera_index: int = 0):
        """Initialize camera.

        Args:
            camera_index: Index of the camera to use (default: 0 for laptop webcam)
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def start(self) -> bool:
        """Start the camera capture.

        Returns:
            True if camera started successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the camera.

        Returns:
            Frame as numpy array, or None if no frame available
        """
        if not self.is_running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the current frame from camera.

        Returns:
            Tuple of (success, frame)
        """
        frame = self.read_frame()
        if frame is None:
            return False, None
        return True, frame

    def stop(self):
        """Stop the camera and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def test_camera() -> bool:
    """Test if camera is accessible.

    Returns:
        True if camera works, False otherwise
    """
    with Camera() as cam:
        return cam.is_running