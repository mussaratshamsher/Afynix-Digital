"""Smart Attendance System - Main Application."""

import cv2
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.attendance_manager import AttendanceManager
from src.utils import draw_label


class AttendanceApp:
    """Main attendance application."""

    def __init__(self, camera_index: int = 0):
        """Initialize the application."""
        self.camera = Camera(camera_index)
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()

        self.is_running = False
        self.show_video = True

        # Track recently recognized to avoid duplicate attendance
        self.last_recognized = {}
        self.cooldown_seconds = 10

    def start(self):
        """Start the attendance system."""
        print("Starting Smart Attendance System...")
        print("Press 'q' to quit")
        print("Press 's' to stop video display")

        # Check if we have registered faces
        if not self.face_recognizer.has_registered_faces():
            print("WARNING: No registered faces found.")
            print("Register users first with: python app.py --register 'Name'")

        # Start camera
        if not self.camera.start():
            print("Error: Could not start camera")
            return False

        self.is_running = True

        # Main loop
        while self.is_running:
            # Read frame
            success, frame = self.camera.get_frame()
            if not success:
                continue

            # Process frame
            processed_frame = self.process_frame(frame)

            # Show video
            if self.show_video:
                cv2.imshow("Smart Attendance System", processed_frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_video = not self.show_video

        # Cleanup
        self.camera.stop()
        cv2.destroyAllWindows()

        return True

    def process_frame(self, frame):
        """Process a single frame."""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)

        # Draw face boxes
        display_frame = self.face_detector.draw_faces(faces, frame.copy())

        # Recognize each face
        if len(faces) > 0 and self.face_recognizer.has_registered_faces():
            # Convert frame to grayscale for recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Recognize faces
            results = self.face_recognizer.recognize_faces(gray, faces)

            for result in results:
                name = result['name']
                confidence = result['confidence']
                x, y, w, h = result['box']

                # Draw name label
                display_frame = draw_label(
                    display_frame,
                    f"{name}",
                    (x, y - 10),
                    bg_color=(0, 255, 0)
                )

                # Mark attendance (with cooldown to prevent duplicates)
                current_time = time.time()
                last_time = self.last_recognized.get(name, 0)

                if current_time - last_time > self.cooldown_seconds:
                    marked = self.attendance_manager.mark_attendance(name)
                    if marked:
                        print(f"Attendance marked: {name}")
                        self.last_recognized[name] = current_time

        return display_frame

    def register_new_face(self, name: str, image_path: str = None):
        """Register a new face."""
        # Start camera if not already running
        if not self.camera.is_running:
            print("Starting camera...")
            if not self.camera.start():
                print("Could not start camera")
                return False

        if image_path:
            # Register from file
            success = self.face_recognizer.register_face(image_path, name)
        else:
            # Capture from camera
            print("Looking for face...")

            for _ in range(30):  # Try for 30 frames
                success, frame = self.camera.get_frame()
                if not success:
                    continue

                # Detect face
                faces = self.face_detector.detect_faces(frame)
                if len(faces) > 0:
                    break

                time.sleep(0.1)

            if len(faces) == 0:
                print("No face detected")
                return False

            # Get first face
            face_box = faces[0]

            # Register using the new method
            success = self.face_recognizer.register_face_box(frame, face_box, name)

        if success:
            print(f"Successfully registered: {name}")
        else:
            print(f"Failed to register: {name}")

        return success

    def stop(self):
        """Stop the application."""
        self.is_running = False
        self.camera.stop()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Smart Attendance System')
    parser.add_argument('--register', '-r', type=str, help='Register a new person')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index')
    parser.add_argument('--list', '-l', action='store_true', help='List registered users')

    args = parser.parse_args()

    app = AttendanceApp(camera_index=args.camera)

    if args.register:
        # Register mode
        app.register_new_face(args.register)
    elif args.list:
        # List mode
        names = app.face_recognizer.get_registered_names()
        print("Registered users:")
        if names:
            for name in names:
                print(f"  - {name}")
        else:
            print("  (none)")
    else:
        # Run mode
        app.start()


if __name__ == '__main__':
    main()