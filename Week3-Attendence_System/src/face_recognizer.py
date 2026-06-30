"""Face recognition using histogram comparison and LBPH (no dlib required)."""

import os
import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class FaceRecognizer:
    """Handles face recognition with multiple samples per person."""

    def __init__(self, faces_dir: Optional[str] = None, model_file: Optional[str] = None):
        """Initialize face recognizer.

        Args:
            faces_dir: Directory containing known face images
            model_file: Path to save/load face histograms
        """
        if faces_dir is None:
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            faces_dir = base_dir / "data" / "faces"

        if model_file is None:
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            model_file = base_dir / "models" / "face_data.pkl"

        self.faces_dir = Path(faces_dir)
        self.model_file = Path(model_file)

        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.model_file.parent.mkdir(parents=True, exist_ok=True)

        # Store face histograms and names - now supports multiple samples per person
        self.face_histograms: List[np.ndarray] = []
        self.face_images: List[np.ndarray] = []
        self.names: List[str] = []
        self.registration_times: List[str] = []

        # LBPH parameters for more robust recognition
        self.lbph_radius = 1
        self.lbph_neighbors = 8
        self.lbph_grid_x = 8
        self.lbph_grid_y = 8

        # Recognition threshold (lower = easier matching)
        self.confidence_threshold = 20

        # Loading statistics
        self.model_info: Dict = {}

        self.load_model()

    def _create_lbph_recognizer(self):
        """Create LBPH face recognizer for better accuracy."""
        return cv2.face.LBPHFaceRecognizer_create(
            radius=self.lbph_radius,
            neighbors=self.lbph_neighbors,
            grid_x=self.lbph_grid_x,
            grid_y=self.lbph_grid_y
        )

    def _histogram_correlation(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute histogram correlation without cv2.compareHist.

        Uses Pearson correlation coefficient between histograms.

        Args:
            hist1: First histogram
            hist2: Second histogram

        Returns:
            Correlation score (-1 to 1, higher = more similar)
        """
        # Normalize histograms
        h1 = hist1.flatten()
        h2 = hist2.flatten()

        # Calculate means
        mean1 = np.mean(h1)
        mean2 = np.mean(h2)

        # Calculate correlation
        if np.std(h1) == 0 or np.std(h2) == 0:
            return 0.0

        covariance = np.mean((h1 - mean1) * (h2 - mean2))
        std_product = np.std(h1) * np.std(h2)

        if std_product == 0:
            return 0.0

        return covariance / std_product

    def _compute_histogram(self, gray_face: np.ndarray) -> np.ndarray:
        """Compute histogram of a grayscale face."""
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_features(self, gray_face: np.ndarray) -> np.ndarray:
        """Compute simple features from face (resize to fixed size)."""
        resized = cv2.resize(gray_face, (50, 50))
        return resized.flatten().astype(np.float32)

    def _preprocess_face(self, gray_face: np.ndarray) -> np.ndarray:
        """Preprocess face for better recognition."""
        # Resize to consistent size
        resized = cv2.resize(gray_face, (100, 100))

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resized = clahe.apply(resized)

        return resized

    def register_face(self, image_path: str, name: str) -> bool:
        """Register a new face from an image file.

        Args:
            image_path: Path to face image
            name: Name of the person

        Returns:
            True if registered successfully, False otherwise
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False

        return self._add_face(image, name)

    def register_face_from_array(self, image_array: np.ndarray, name: str) -> bool:
        """Register a face from image array.

        Args:
            image_array: Face image as numpy array (grayscale)
            name: Name of the person

        Returns:
            True if registered successfully, False otherwise
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        return self._add_face(gray, name)

    def register_face_box(self, frame: np.ndarray, face_box: Tuple[int, int, int, int],
                      name: str, add_multiple: bool = True) -> bool:
        """Register a face from a bounding box in a frame.

        Args:
            frame: Full image frame (BGR)
            face_box: Face bounding box (x, y, w, h)
            name: Name of the person
            add_multiple: Whether to add multiple samples for better recognition

        Returns:
            True if registered successfully, False otherwise
        """
        x, y, w, h = face_box

        # Extract face region
        face_roi = frame[y:y+h, x:x+w]

        # Handle empty/None ROI or grayscale input
        if face_roi is None or face_roi.size == 0:
            return False

        # Convert to grayscale if needed (handle both BGR and grayscale frames)
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        elif len(face_roi.shape) == 2:
            gray = face_roi
        else:
            return False

        # Preprocess face
        processed = self._preprocess_face(gray)

        # Save the main face image
        face_path = self.faces_dir / f"{name}.jpg"
        cv2.imwrite(str(face_path), processed)

        # Add to recognition model
        success = self._add_face(processed, name)

        # Optionally add more samples for better recognition
        if add_multiple and success:
            # Add slightly rotated variants for robustness
            self._add_augmented_samples(frame, face_box, name)

        return success

    def _add_augmented_samples(self, frame: np.ndarray, face_box: Tuple[int, int, int, int], name: str):
        """Add augmented samples for more robust recognition."""
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]

        # Handle empty/None ROI
        if face_roi is None or face_roi.size == 0:
            return

        # Convert to grayscale if needed
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        elif len(face_roi.shape) == 2:
            gray = face_roi
        else:
            return

        # Get existing count for this person
        existing_count = sum(1 for n in self.names if n == name)

        # Add up to 3 more samples
        if existing_count >= 3:
            return

        # Create flipped version
        flipped = cv2.flip(gray, 1)
        flipped_processed = self._preprocess_face(flipped)
        self._add_face(flipped_processed, name, save_image=False)

    def _add_face(self, gray_face: np.ndarray, name: str, save_image: bool = True) -> bool:
        """Add a face to the recognition model."""
        # Preprocess
        processed = self._preprocess_face(gray_face)

        # Resize
        resized = cv2.resize(processed, (100, 100))

        # Compute histogram
        hist = self._compute_histogram(resized)

        # Compute features
        features = self._compute_features(resized)

        # Add to model
        self.face_histograms.append(hist)
        self.face_images.append(features)

        # Handle multiple registrations of same person
        base_name = name
        count = sum(1 for n in self.names if n.startswith(name))
        if count > 0:
            display_name = f"{name}_{count}"
        else:
            display_name = name

        self.names.append(display_name)
        self.registration_times.append(datetime.now().isoformat())

        self.save_model()
        return True

    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face using multiple comparison methods.

        Args:
            face_image: Face image as numpy array (grayscale)

        Returns:
            Tuple of (name, confidence). Name is None if unknown.
            Higher confidence is better (0-100)
        """
        if len(self.names) == 0:
            return None, 0.0

        # Preprocess face
        processed = self._preprocess_face(face_image)

        # Resize
        resized = cv2.resize(processed, (100, 100))

        # Compute histogram
        hist = self._compute_histogram(resized)

        # Compute features
        features = self._compute_features(resized)

        # Group by person and compute best match for each
        person_scores: Dict[str, List[float]] = {}

        for i, (stored_hist, stored_features, name) in enumerate(zip(
            self.face_histograms, self.face_images, self.names
        )):
            # Extract base name (remove _1, _2 suffix if present)
            base_name = name.rsplit('_', 1)[0] if '_' in name and name.split('_')[-1].isdigit() else name
            if base_name not in person_scores:
                person_scores[base_name] = []

            # Compare histograms using custom correlation (no cv2 dependency)
            hist_score = self._histogram_correlation(hist, stored_hist)
            hist_similarity = max(0, hist_score * 100)  # Convert to 0-100 similar

            # Compare features (correlation - higher is more similar)
            if np.std(features) > 0 and np.std(stored_features) > 0:
                corr = np.corrcoef(features, stored_features)[0, 1]
                if np.isnan(corr):
                    corr = 0
                features_similarity = (corr + 1) * 50  # Convert -1:1 to 0-100
            else:
                features_similarity = 0

            # Combined score
            score = hist_similarity * 0.3 + features_similarity * 0.7
            person_scores[base_name].append(score)

        # Get best score for each person and find overall best
        best_match = None
        best_score = 0.0

        for person, scores in person_scores.items():
            # Use average of all samples for this person
            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_match = person

        if best_match and best_score > self.confidence_threshold:
            return best_match, best_score

        return None, 0.0

    def recognize_faces(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Recognize all faces in a frame.

        Args:
            frame: Image frame (BGR)
            face_boxes: List of face bounding boxes (x, y, w, h)

        Returns:
            List of dicts with name, confidence, and location
        """
        results = []

        for x, y, w, h in face_boxes:
            # Extract face
            face_roi = frame[y:y+h, x:x+w]

            # Handle empty/None ROI
            if face_roi is None or face_roi.size == 0:
                continue

            # Convert to grayscale if needed
            if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            elif len(face_roi.shape) == 2:
                gray = face_roi
            else:
                continue

            # Recognize
            name, confidence = self.recognize_face(gray)

            if name is not None:
                results.append({
                    "name": name,
                    "confidence": confidence,
                    "box": (x, y, w, h)
                })

        return results

    def recognize_with_details(self, face_image: np.ndarray) -> Dict:
        """Recognize face with detailed results.

        Args:
            face_image: Face image as numpy array

        Returns:
            Dictionary with recognition details
        """
        if len(self.names) == 0:
            return {
                "recognized": False,
                "name": None,
                "confidence": 0.0,
                "alternative_names": []
            }

        processed = self._preprocess_face(face_image)
        resized = cv2.resize(processed, (100, 100))

        hist = self._compute_histogram(resized)
        features = self._compute_features(resized)

        # Get all scores
        all_scores = []
        for stored_hist, stored_features, name in zip(
            self.face_histograms, self.face_images, self.names
        ):
            base_name = name.rsplit('_', 1)[0] if '_' in name and name.split('_')[-1].isdigit() else name

            hist_score = self._histogram_correlation(hist, stored_hist)
            hist_similarity = max(0, hist_score * 100)

            if np.std(features) > 0 and np.std(stored_features) > 0:
                corr = np.corrcoef(features, stored_features)[0, 1]
                if np.isnan(corr):
                    corr = 0
                features_similarity = (corr + 1) * 50
            else:
                features_similarity = 0

            score = hist_similarity * 0.3 + features_similarity * 0.7
            all_scores.append((base_name, score))

        # Sort by score
        all_scores.sort(key=lambda x: x[1], reverse=True)

        # Get best match
        if all_scores and all_scores[0][1] > self.confidence_threshold:
            return {
                "recognized": True,
                "name": all_scores[0][0],
                "confidence": all_scores[0][1],
                "alternative_names": [
                    {"name": s[0], "confidence": s[1]}
                    for s in all_scores[1:3] if s[1] > self.confidence_threshold
                ]
            }

        return {
            "recognized": False,
            "name": None,
            "confidence": 0.0,
            "alternative_names": []
        }

    def load_model(self):
        """Load saved model from file."""
        if not self.model_file.exists():
            return

        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
                self.face_histograms = data.get('histograms', [])
                self.face_images = data.get('images', [])
                self.names = data.get('names', [])
                self.registration_times = data.get('times', [])
                self.model_info = data.get('info', {})
        except Exception as e:
            print(f"Error loading model: {e}")

    def save_model(self):
        """Save model to file."""
        data = {
            'histograms': self.face_histograms,
            'images': self.face_images,
            'names': self.names,
            'times': self.registration_times,
            'info': {
                'saved_at': datetime.now().isoformat(),
                'total_faces': len(self.names),
                'unique_persons': len(set(n.rsplit('_', 1)[0] for n in self.names))
            }
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(data, f)

    def get_registered_names(self) -> List[str]:
        """Get list of unique registered names."""
        unique_names = []
        for name in self.names:
            base_name = name.rsplit('_', 1)[0] if '_' in name and name.split('_')[-1].isdigit() else name
            if base_name not in unique_names:
                unique_names.append(base_name)
        return unique_names

    def get_registration_count(self, name: str) -> int:
        """Get number of samples for a person."""
        return sum(1 for n in self.names if n.startswith(name))

    def clear_person(self, name: str) -> bool:
        """Remove a person's face data.

        Args:
            name: Name of person to remove

        Returns:
            True if person was removed
        """
        if name not in self.get_registered_names():
            return False

        # Remove all samples for this person
        new_histograms = []
        new_images = []
        new_names = []
        new_times = []

        for hist, img, n, t in zip(
            self.face_histograms, self.face_images,
            self.names, self.registration_times
        ):
            base = n.rsplit('_', 1)[0] if '_' in n and n.split('_')[-1].isdigit() else n
            if base != name:
                new_histograms.append(hist)
                new_images.append(img)
                new_names.append(n)
                new_times.append(t)

        self.face_histograms = new_histograms
        self.face_images = new_images
        self.names = new_names
        self.registration_times = new_times

        # Delete face image file
        face_path = self.faces_dir / f"{name}.jpg"
        if face_path.exists():
            face_path.unlink()

        self.save_model()
        return True

    def clear_all(self):
        """Clear all registered faces."""
        self.face_histograms = []
        self.face_images = []
        self.names = []
        self.registration_times = []

        # Delete model file
        if self.model_file.exists():
            self.model_file.unlink()

        # Delete face images
        for face_file in self.faces_dir.glob("*.jpg"):
            face_file.unlink()

    def has_registered_faces(self) -> bool:
        """Check if there are registered faces."""
        return len(self.names) > 0

    def get_model_info(self) -> Dict:
        """Get model information."""
        unique_names = self.get_registered_names()
        return {
            "total_samples": len(self.names),
            "unique_persons": len(unique_names),
            "persons": unique_names,
            "saved_at": self.model_info.get("saved_at"),
            "last_updated": self.registration_times[-1] if self.registration_times else None
        }

    def set_confidence_threshold(self, threshold: float):
        """Set recognition confidence threshold.

        Args:
            threshold: New threshold (0-100, lower = more strict)
        """
        self.confidence_threshold = max(0, min(100, threshold))

    def is_duplicate_face(self, face_image: np.ndarray, threshold: float = 70.0) -> Tuple[Optional[str], float]:
        """Check if face already exists in database.

        Args:
            face_image: Face image to check
            threshold: Similarity threshold to consider duplicate (0-100)

        Returns:
            Tuple of (existing_name, similarity) if duplicate found, else (None, 0.0)
        """
        if len(self.names) == 0:
            return None, 0.0

        # Preprocess face
        processed = self._preprocess_face(face_image)
        resized = cv2.resize(processed, (100, 100))
        hist = self._compute_histogram(resized)
        features = self._compute_features(resized)

        # Compare with all stored faces
        best_match = None
        best_score = 0.0

        for stored_hist, stored_features, name in zip(
            self.face_histograms, self.face_images, self.names
        ):
            # Extract base name
            base_name = name.rsplit('_', 1)[0] if '_' in name and name.split('_')[-1].isdigit() else name

            # Compare histograms
            hist_score = self._histogram_correlation(hist, stored_hist)
            hist_similarity = max(0, hist_score * 100)

            # Compare features
            if np.std(features) > 0 and np.std(stored_features) > 0:
                corr = np.corrcoef(features, stored_features)[0, 1]
                if np.isnan(corr):
                    corr = 0
                features_similarity = (corr + 1) * 50
            else:
                features_similarity = 0

            # Combined score
            score = hist_similarity * 0.3 + features_similarity * 0.7

            if score > best_score:
                best_score = score
                best_match = base_name

        if best_match and best_score >= threshold:
            return best_match, best_score

        return None, 0.0