"""Face detection using Haar Cascade Classifier."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class FaceDetector:
    """Detects faces in images using Haar Cascade Classifier."""

    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize face detector.

        Args:
            cascade_path: Path to Haar Cascade XML file
        """
        if cascade_path is None:
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            cascade_path = base_dir / "models" / "haarcascade_frontalface_default.xml"

        self.cascade = cv2.CascadeClassifier(str(cascade_path))

        if self.cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")

        # Detection parameters - optimized for better detection
        self.scale_factor = 1.05  # Smaller step for more precise detection
        self.min_neighbors = 3  # Lower threshold for more detections
        self.min_size = (30, 30)

        # Adaptive detection settings
        self.equalize_histogram = True  # Improve contrast for lighting
        self.detection_mode = "auto"  # auto, strict, fast

    def _apply_adaptive_lighting(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive lighting compensation for different conditions.

        Args:
            gray: Grayscale image

        Returns:
            Processed grayscale image
        """
        if self.equalize_histogram:
            # CLAHE for better contrast in varying lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        return gray

    def _detect_with_multi_scale(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Multi-scale face detection for better accuracy.

        Args:
            gray: Grayscale image

        Returns:
            List of face bounding boxes
        """
        # Preprocess with adaptive lighting
        processed = self._apply_adaptive_lighting(gray)

        # Detect with standard parameters
        faces = self.cascade.detectMultiScale(
            processed,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        if faces is None or len(faces) == 0:
            return []

        # Convert to list of tuples
        faces_arr = np.array(faces)
        if faces_arr.ndim == 1:
            return []

        return [tuple(int(v) for v in row) for row in faces_arr]

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.

        Args:
            frame: Input image frame

        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply lighting compensation
        if self.equalize_histogram:
            gray = self._apply_adaptive_lighting(gray)

        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        # Handle empty result - OpenCV returns different shapes
        if faces is None or len(faces) == 0:
            return []

        # Force convert to 2D array and then to list of tuples
        faces_arr = np.array(faces)
        if faces_arr.ndim == 1:
            return []

        return [tuple(int(v) for v in row) for row in faces_arr]

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.

        Args:
            frame: Input image frame

        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        # Handle empty result - OpenCV returns different shapes
        if faces is None or len(faces) == 0:
            return []

        # Force convert to 2D array and then to list of tuples
        faces_arr = np.array(faces)
        if faces_arr.ndim == 1:
            return []

        return [tuple(int(v) for v in row) for row in faces_arr]

    def draw_faces(self, faces, frame, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes around detected faces."""
        result = frame.copy()
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(result, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), color, int(thickness))
        return result

    def draw_faces_with_confidence(self, faces: List[Tuple], frame: np.ndarray,
                                   confidences: List[float] = None,
                                   thickness: int = 2) -> np.ndarray:
        """Draw faces with confidence indicators.

        Args:
            faces: List of face bounding boxes
            frame: Input frame
            confidences: List of confidence scores (0-100)
            thickness: Box thickness

        Returns:
            Frame with drawn faces
        """
        result = frame.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face
            # Color based on confidence (green=high, yellow=medium, red=low)
            if confidences and i < len(confidences):
                conf = confidences[i]
                if conf > 70:
                    color = (0, 255, 0)  # Green
                elif conf > 40:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)

            cv2.rectangle(result, (int(x), int(y)),
                         (int(x) + int(w), int(y) + int(h)),
                         color, thickness)

        return result

    def get_face_roi(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region of interest.

        Args:
            frame: Input image frame
            face_box: Face bounding box (x, y, w, h)

        Returns:
            Face region as image
        """
        x, y, w, h = face_box
        return frame[y:y+h, x:x+w]

    def assess_lighting_conditions(self, frame: np.ndarray) -> Dict[str, any]:
        """Assess lighting conditions for testing.

        Args:
            frame: Input frame

        Returns:
            Dictionary with lighting metrics
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Calculate metrics
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)

        # Determine condition
        if mean_brightness > 100 and std_dev > 50:
            condition = "good"
        elif mean_brightness > 60:
            condition = "fair"
        else:
            condition = "poor"

        return {
            "mean_brightness": float(mean_brightness),
            "std_deviation": float(std_dev),
            "condition": condition,
            "recommendation": self._get_lighting_recommendation(condition)
        }

    def _get_lighting_recommendation(self, condition: str) -> str:
        """Get recommendation based on lighting condition."""
        recommendations = {
            "good": "Lighting is optimal for face detection",
            "fair": "Lighting is acceptable but could be improved",
            "poor": "Improve lighting for better detection"
        }
        return recommendations.get(condition, "Unknown condition")

    def set_detection_mode(self, mode: str):
        """Set detection mode.

        Args:
            mode: 'auto', 'strict' (fewer false positives), 'fast' (speed priority)
        """
        self.detection_mode = mode
        if mode == "strict":
            self.min_neighbors = 5
            self.scale_factor = 1.2
        elif mode == "fast":
            self.min_neighbors = 4
            self.scale_factor = 1.15
        else:  # auto
            self.min_neighbors = 3
            self.scale_factor = 1.05