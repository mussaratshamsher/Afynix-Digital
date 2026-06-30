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

        # Detection parameters - optimized for better detection with lower min_neighbors
        self.scale_factor = 1.1  # Faster detection
        self.min_neighbors = 3  # Lower for better detection of侧面/角度
        self.min_size = (50, 50)  # Larger min size for better quality

        # Adaptive detection settings
        self.equalize_histogram = True  # Enable for better detection
        self.detection_mode = "auto"  # Auto mode for best detection

    def _apply_adaptive_lighting(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive lighting compensation for different conditions."""
        if self.equalize_histogram:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        return gray

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

        # Apply lighting compensation for better detection
        gray = self._apply_adaptive_lighting(gray)

        # Detect faces with optimized parameters
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        # Handle empty result
        if faces is None or len(faces) == 0:
            return []

        # Convert to list of tuples
        faces_arr = np.array(faces)
        if faces_arr.ndim == 1:
            return []

        return [tuple(int(v) for v in row) for row in faces_arr]

    def detect_best_face(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Dict]:
        """Detect the best quality face in frame with quality assessment.

        Args:
            frame: Input image frame

        Returns:
            Tuple of (best_face_box, quality_info)
        """
        faces = self.detect_faces(frame)

        if not faces:
            return None, {"quality": "no_face", "message": "No face detected"}

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Score each face by quality (size, centered, visibility)
        best_face = None
        best_score = -1
        best_info = {}

        for face_box in faces:
            x, y, w, h = face_box
            score = 0
            issues = []
            messages = []

            # Score 1: Face size (larger = better quality)
            size_score = min(w * h / (frame_width * frame_height) * 100, 30)
            score += size_score

            # Score 2: Face centered in frame
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            frame_center_x = frame_width // 2
            frame_center_y = frame_height // 2

            dev_x = abs(face_center_x - frame_center_x) / frame_width
            dev_y = abs(face_center_y - frame_center_y) / frame_height

            center_score = max(0, 20 - (dev_x + dev_y) * 50)
            score += center_score

            if dev_x > 0.2:
                issues.append("not_centered")
                messages.append("Move face to center")
            if dev_y > 0.2:
                issues.append("not_centered")
                messages.append("Move face to center")

            # Score 3: Face brightness/contrast (visibility)
            face_roi = frame[y:y+h, x:x+w]
            if len(face_roi.shape) == 3:
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_roi

            mean_brightness = np.mean(face_gray)
            std_contrast = np.std(face_gray)

            # Good visibility: brightness 60-180, contrast > 40
            if 60 <= mean_brightness <= 180 and std_contrast > 40:
                score += 25
            elif mean_brightness < 50:
                score -= 10
                issues.append("too_dark")
                messages.append("Improve lighting - too dark")
            elif mean_brightness > 200:
                score -= 10
                issues.append("too_bright")
                messages.append("Reduce lighting - too bright")
            elif std_contrast < 30:
                score -= 5
                issues.append("low_contrast")
                messages.append("Improve lighting - low contrast")

            # Score 4: Face symmetry (for angled faces)
            # Check left/right half similarity
            half_w = w // 2
            left_half = face_gray[:, :half_w] if len(face_gray.shape) == 2 else face_gray[:, :half_w, :]
            right_half = cv2.flip(face_gray[:, half_w:] if len(face_gray.shape) == 2 else face_gray[:, half_w:, :], 1)

            if left_half.size > 0 and right_half.size > 0:
                left_half = cv2.resize(left_half, (50, 50))
                right_half = cv2.resize(right_half, (50, 50))
                correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]

                if not np.isnan(correlation):
                    # Front-facing: high correlation (> 0.5)
                    # Side-facing: lower correlation
                    if correlation > 0.6:
                        score += 25
                    elif correlation > 0.4:
                        score += 15
                    else:
                        score += 5
                        issues.append("angled")
                        messages.append("Face not straight - look at camera")

            # Track best face
            if score > best_score:
                best_score = score
                best_face = face_box
                best_info = {
                    "quality": "good" if score > 50 else "fair" if score > 30 else "poor",
                    "score": float(score),
                    "issues": issues,
                    "messages": messages,
                    "size": (w, h),
                    "centered": len(issues) == 0 or "not_centered" not in issues,
                    "brightness": float(mean_brightness),
                    "contrast": float(std_contrast)
                }

        return best_face, best_info

    def assess_face_quality(self, frame: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict:
        """Assess the quality of a detected face.

        Args:
            frame: Input image frame
            face_box: Face bounding box (x, y, w, h)

        Returns:
            Dictionary with quality assessment
        """
        x, y, w, h = face_box
        frame_height, frame_width = frame.shape[:2]

        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return {"valid": False, "message": "Invalid face region"}

        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi

        # Check brightness
        mean_brightness = np.mean(face_gray)
        std_contrast = np.std(face_gray)

        # Check size
        face_area = w * h
        frame_area = frame_width * frame_height
        size_ratio = face_area / frame_area

        # Check position
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        dev_x = abs(face_center_x - frame_center_x) / frame_width
        dev_y = abs(face_center_y - frame_center_y) / frame_height

        # Build validation result
        issues = []
        messages = []

        # Size check
        if size_ratio < 0.06:
            issues.append("too_small")
            messages.append("Get closer to camera")
        elif size_ratio > 0.5:
            issues.append("too_large")
            messages.append("Move back slightly")

        # Brightness check
        if mean_brightness < 50:
            issues.append("too_dark")
            messages.append("Improve lighting - too dark")
        elif mean_brightness > 200:
            issues.append("too_bright")
            messages.append("Reduce lighting - too bright")

        # Contrast check
        if std_contrast < 30:
            issues.append("low_contrast")
            messages.append("Improve lighting - low contrast")

        # Position check
        if dev_x > 0.2 or dev_y > 0.2:
            issues.append("not_centered")
            messages.append("Center your face")

        # Minimum size check
        if w < 60 or h < 60:
            issues.append("too_small")
            messages.append("Get closer - face too small")

        is_valid = len(issues) == 0

        return {
            "valid": is_valid,
            "message": messages[0] if messages else "OK",
            "issues": issues,
            "messages": messages,
            "size_ratio": size_ratio,
            "brightness": mean_brightness,
            "contrast": std_contrast,
            "centered": dev_x < 0.2 and dev_y < 0.2
        }

    def validate_face_for_registration(self, face_box: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[bool, str]:
        """Validate if face is good enough for registration.

        Args:
            face_box: Face bounding box (x, y, w, h)
            frame_width: Total frame width
            frame_height: Total frame height

        Returns:
            Tuple of (is_valid, message)
        """
        x, y, w, h = face_box

        # Check minimum size
        if w < 60 or h < 60:
            return False, "Face too small - get closer"

        # Check if face is centered
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        dev_x = abs(face_center_x - frame_center_x) / frame_width
        dev_y = abs(face_center_y - frame_center_y) / frame_height

        if dev_x > 0.2:
            if face_center_x < frame_center_x:
                return False, "Move face slightly to the right"
            else:
                return False, "Move face slightly to the left"

        if dev_y > 0.2:
            if face_center_y < frame_center_y:
                return False, "Move face slightly lower"
            else:
                return False, "Move face slightly higher"

        return True, "Face is properly positioned"

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
        """Draw faces with confidence indicators."""
        result = frame.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face
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
        """Extract face region of interest."""
        x, y, w, h = face_box
        return frame[y:y+h, x:x+w]

    def assess_lighting_conditions(self, frame: np.ndarray) -> Dict[str, any]:
        """Assess lighting conditions."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)

        if mean_brightness > 100 and std_dev > 50:
            condition = "good"
        elif mean_brightness > 60:
            condition = "fair"
        else:
            condition = "poor"

        recommendations = {
            "good": "Lighting is optimal",
            "fair": "Lighting is acceptable",
            "poor": "Improve lighting"
        }

        return {
            "mean_brightness": float(mean_brightness),
            "std_deviation": float(std_dev),
            "condition": condition,
            "recommendation": recommendations.get(condition, "Unknown")
        }

    def set_detection_mode(self, mode: str):
        """Set detection mode."""
        self.detection_mode = mode
        if mode == "strict":
            self.min_neighbors = 5
            self.scale_factor = 1.2
        elif mode == "fast":
            self.min_neighbors = 3
            self.scale_factor = 1.15
        else:  # auto
            self.min_neighbors = 3
            self.scale_factor = 1.1