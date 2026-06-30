"""Smart Attendance System package."""

from .camera import Camera
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .attendance_manager import AttendanceManager

__all__ = [
    'Camera',
    'FaceDetector',
    'FaceRecognizer',
    'AttendanceManager',
]