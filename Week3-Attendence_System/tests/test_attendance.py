"""Tests for attendance system."""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attendance_manager import AttendanceManager
from src.camera import Camera
from src.face_detector import FaceDetector


class TestAttendanceManager:
    """Test attendance manager."""

    def test_init(self):
        """Test initialization."""
        manager = AttendanceManager()
        assert manager is not None
        assert manager.attendance_file is not None

    def test_mark_attendance(self):
        """Test marking attendance."""
        manager = AttendanceManager()
        name = "TestUser"

        # Mark attendance
        marked = manager.mark_attendance(name)
        assert marked is True

        # Check duplicate prevention
        marked = manager.mark_attendance(name)
        assert marked is False


class TestCamera:
    """Test camera module."""

    def test_init(self):
        """Test initialization."""
        camera = Camera()
        assert camera.camera_index == 0
        assert not camera.is_running


class TestFaceDetector:
    """Test face detector."""

    def test_init(self):
        """Test initialization."""
        detector = FaceDetector()
        assert detector is not None
        assert detector.cascade is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])