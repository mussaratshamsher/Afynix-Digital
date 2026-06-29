"""Enhanced Streamlit UI for Smart Attendance System."""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, date
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera import Camera
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.attendance_manager import AttendanceManager


class StreamlitAttendanceApp:
    """Streamlit-based attendance application with enhanced features."""

    def __init__(self):
        """Initialize the app."""
        # Initialize components
        self.camera = Camera()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()

        # Session state for settings
        if 'detection_mode' not in st.session_state:
            st.session_state.detection_mode = 'auto'
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 30
        if 'equalize_histogram' not in st.session_state:
            st.session_state.equalize_histogram = True
        if 'attendance_running' not in st.session_state:
            st.session_state.attendance_running = False
        if 'registration_name' not in st.session_state:
            st.session_state.registration_name = ''
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'

    def _inject_theme_css(self):
        """Inject custom theme CSS."""
        is_dark = st.session_state.theme == 'dark'
        bg = "#0E1117" if is_dark else "#FFFFFF"
        sidebar_bg = "#262730" if is_dark else "#F0F2F6"
        text = "#FAFAFA" if is_dark else "#31333F"
        primary = "#00E676"
        secondary = "#31333F" if is_dark else "#F0F2F6"

        st.markdown(f"""
            <style>
            /* Main app background */
            .stApp {{
                background-color: {bg};
            }}
            /* Text colors */
            .stMarkdown, .stText, p, span, li, div[data-testid="stMetricValue"] {{
                color: {text} !important;
            }}
            /* Headings */
            h1, h2, h3, h4 {{
                color: {text} !important;
            }}
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: {sidebar_bg};
            }}
            section[data-testid="stSidebar"] * {{
                color: {text} !important;
            }}
            /* Buttons */
            .stButton > button {{
                background-color: {primary};
                color: #000000;
                border: none;
            }}
            .stButton > button:hover {{
                background-color: #00C853;
            }}
            /* Cards/containers */
            div[data-testid="stMetricCard"] {{
                background-color: {sidebar_bg};
                color: {text};
            }}
            /* Input fields */
            .stTextInput > div > div > input {{
                background-color: {secondary};
                color: {text};
            }}
            /* Dataframes */
            .stDataFrame {{
                background-color: {bg};
            }}
            </style>
        """, unsafe_allow_html=True)

    
    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(
            page_title="Smart Attendance System",
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply session state to components
        self.face_detector.equalize_histogram = st.session_state.equalize_histogram
        self.face_detector.set_detection_mode(st.session_state.detection_mode)
        self.face_recognizer.set_confidence_threshold(st.session_state.confidence_threshold)

        # Inject theme CSS
        self._inject_theme_css()

        # Title with theme toggle
        col_title, col_toggle = st.columns([6, 1])
        with col_title:
            st.title("Smart Attendance System")
        with col_toggle:
            st.write("")
            st.write("")
            icon = "🌙" if st.session_state.theme == 'light' else "☀️"
            if st.button(icon, help="Toggle Dark/Light Mode"):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.rerun()

        st.markdown("---")

        # Sidebar with settings
        self.sidebar()

        # Main content
        page = st.sidebar.radio("Go to", ["Home", "Register Face", "View Attendance", "Statistics", "Settings"])

        if page == "Home":
            self.home_page()
        elif page == "Register Face":
            self.register_page()
        elif page == "View Attendance":
            self.attendance_page()
        elif page == "Statistics":
            self.statistics_page()
        elif page == "Settings":
            self.settings_page()

    def sidebar(self):
        """Render sidebar with menu and settings."""
        st.sidebar.title("Menu")

        # Model info
        with st.sidebar.expander("Model Info", expanded=False):
            info = self.face_recognizer.get_model_info()
            st.write(f"**Total Samples:** {info['total_samples']}")
            st.write(f"**Unique Persons:** {info['unique_persons']}")
            if info['persons']:
                st.write("**Persons:**")
                for p in info['persons']:
                    st.write(f"  - {p}")

    def home_page(self):
        """Home page with start/stop buttons and live view."""
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Live Camera Feed")

            # Video placeholder
            video_placeholder = st.empty()
            status_placeholder = st.empty()

            # Controls
            col_start, col_stop = st.columns(2)

            with col_start:
                if st.button("Start Attendance", type="primary",
                          disabled=st.session_state.attendance_running):
                    if self.camera.start():
                        st.session_state.attendance_running = True
                        status_placeholder.success("Attendance system started!")
                    else:
                        st.error("Could not start camera")

            with col_stop:
                if st.button("Stop Attendance",
                          disabled=not st.session_state.attendance_running):
                    if st.session_state.attendance_running:
                        self.camera.stop()
                        st.session_state.attendance_running = False
                        status_placeholder.info("Attendance system stopped!")

            # Live video loop
            if st.session_state.attendance_running:
                try:
                    while st.session_state.attendance_running:
                        success, frame = self.camera.get_frame()
                        if success:
                            # Assess lighting
                            lighting = self.face_detector.assess_lighting_conditions(frame)

                            # Detect and recognize faces
                            faces = self.face_detector.detect_faces(frame)
                            display_frame = self.face_detector.draw_faces(faces, frame.copy())

                            # If we have registered faces, do recognition
                            names = []
                            confidences = []
                            if self.face_recognizer.has_registered_faces():
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                results = self.face_recognizer.recognize_faces(gray, faces)

                                for result in results:
                                    names.append(result['name'])
                                    confidences.append(result['confidence'])

                                    # Mark attendance
                                    marked = self.attendance_manager.mark_attendance(result['name'])
                                    if marked:
                                        status_placeholder.success(f"Attendance marked: {result['name']}")

                                # Draw with confidence colors
                                display_frame = self.face_detector.draw_faces_with_confidence(
                                    faces, display_frame, confidences
                                )

                                # Add labels
                                from src.utils import draw_label
                                for result in results:
                                    display_frame = draw_label(
                                        display_frame,
                                        f"{result['name']} ({result['confidence']:.0f}%)",
                                        (result['box'][0], result['box'][1] - 10),
                                        bg_color=(0, 255, 0) if result['confidence'] > 70 else (0, 255, 255)
                                    )

                            # Add info overlay
                            from src.utils import add_status_bar
                            info_text = f"Detected: {len(faces)} faces | Lighting: {lighting['condition']} | Press Stop to quit"
                            display_frame = add_status_bar(display_frame, info_text, "bottom")

                            # Convert for display
                            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                            # Display frame
                            video_placeholder.image(display_frame, channels="RGB")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.attendance_running = False

        with col2:
            st.subheader("Today's Attendance")
            attendance = self.attendance_manager.get_today_attendance()

            if attendance:
                df = pd.DataFrame(attendance)
                st.dataframe(df, hide_index=True, height=200)
                st.write(f"**Total Present:** {len(attendance)}")
            else:
                st.info("No attendance recorded today")

            # Quick stats
            st.markdown("---")
            st.subheader("Quick Stats")
            stats = self.attendance_manager.get_statistics(days=7)
            if stats:
                st.write(f"Week Total: {stats['total_attendance']}")
                st.write(f"Daily Average: {stats['daily_average']}")
                st.write(f"Unique Persons: {stats['unique_persons']}")

    def register_page(self):
        """Register a new face with enhanced features."""
        st.subheader("Register New Person")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            name = st.text_input("Enter Name", key="reg_name_input")

        with col2:
            st.write("")
            st.write("")
            multiple = st.checkbox("Add Multiple Samples", value=True,
                            help="Add more samples for better recognition")

        with col3:
            st.write("")
            st.write("")
            use_camera = st.checkbox("Use Live Camera", value=True)

        # Capture area
        col_capture, col_preview = st.columns(2)

        with col_capture:
            if st.button("Capture & Register", type="primary"):
                if not name:
                    st.error("Please enter a name")
                elif use_camera:
                    # Start camera
                    if self.camera.start():
                        st.info("Looking for face...")

                        # Get frame
                        success, frame = self.camera.get_frame()
                        if success:
                            # Detect face
                            faces = self.face_detector.detect_faces(frame)

                            if len(faces) == 0:
                                st.error("No face detected. Please try again.")
                            else:
                                # Get first face box
                                face_box = faces[0]

                                # Register
                                success = self.face_recognizer.register_face_box(
                                    frame, face_box, name,
                                    add_multiple=multiple
                                )

                                if success:
                                    st.success(f"Successfully registered: {name}")
                                    # Feedback
                                    count = self.face_recognizer.get_registration_count(name)
                                    st.info(f"Total samples for {name}: {count}")
                                else:
                                    st.error("Failed to register face")

                        self.camera.stop()
                    else:
                        st.error("Could not start camera")

        with col_preview:
            st.subheader("Preview")
            # Show registered face
            import os
            face_path = Path(__file__).parent.parent / "data" / "faces" / f"{name}.jpg"
            if face_path.exists():
                face_img = cv2.imread(str(face_path))
                # Handle grayscale (1-channel) images
                if len(face_img.shape) == 2 or face_img.shape[2] == 1:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                else:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                st.image(face_img, caption=f"{name}", use_container_width=True)
            else:
                st.info("No preview available")

        st.markdown("---")

        # List registered users with management
        col_list, col_manage = st.columns([2, 1])

        with col_list:
            st.subheader("Registered Users")
            names = self.face_recognizer.get_registered_names()

            if names:
                df = pd.DataFrame({"Name": names})
                # Add sample count
                counts = [self.face_recognizer.get_registration_count(n) for n in names]
                df["Samples"] = counts
                st.dataframe(df, hide_index=True)

                # Info
                st.write(f"**Total:** {len(names)} registered")
            else:
                st.info("No users registered yet")

        with col_manage:
            st.subheader("Manage")
            selected_to_delete = st.selectbox("Select person to remove", [""] + names)

            if st.button("Remove Person"):
                if selected_to_delete:
                    if self.face_recognizer.clear_person(selected_to_delete):
                        st.success(f"Removed: {selected_to_delete}")
                        st.rerun()
                    else:
                        st.error("Failed to remove person")

            if st.button("Clear All"):
                if st.confirm("Are you sure you want to clear all registered faces?"):
                    self.face_recognizer.clear_all()
                    st.success("Cleared all faces")
                    st.rerun()

    def attendance_page(self):
        """View attendance records with filters."""
        st.subheader("Attendance Records")

        # Tabs for different views
        tab_today, tab_date, tab_person, tab_export = st.tabs(
            ["Today", "By Date", "By Person", "Export"]
        )

        with tab_today:
            attendance = self.attendance_manager.get_today_attendance()

            if attendance:
                df = pd.DataFrame(attendance)
                st.dataframe(df, hide_index=True)

                col_count, col_download = st.columns(2)
                with col_count:
                    st.write(f"**Total Present:** {len(attendance)}")

                with col_download:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Today",
                        data=csv,
                        file_name=f"attendance_{date.today()}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No attendance recorded today")

        with tab_date:
            selected_date = st.date_input("Select Date", value=datetime.now().date())

            if selected_date:
                date_str = selected_date.strftime('%Y-%m-%d')
                attendance = self.attendance_manager.get_attendance_by_date(date_str)

                if attendance:
                    df = pd.DataFrame(attendance)
                    st.dataframe(df, hide_index=True)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"attendance_{date_str}.csv",
                        mime="text/csv"
                    )

                    st.write(f"**Total:** {len(attendance)}")
                else:
                    st.info(f"No attendance on {date_str}")

        with tab_person:
            names = self.attendance_manager.get_all_attendance()
            unique_names = list(set(r['Name'] for r in names))

            if unique_names:
                selected_person = st.selectbox("Select Person", unique_names)
                history = self.attendance_manager.get_person_history(selected_person)

                if history:
                    df = pd.DataFrame(history)
                    st.dataframe(df, hide_index=True)

                    # Stats for this person
                    stats = self.attendance_manager.get_person_statistics(selected_person)
                    st.write(f"**Total Days:** {stats['total_days']}")
                    st.write(f"**First Seen:** {stats['first_seen']}")
                    st.write(f"**Last Seen:** {stats['last_seen']}")
                    st.write(f"**Current Streak:** {stats['streak']} days")

                    # Remove today's record
                    if stats['last_seen'] == date.today().isoformat():
                        if st.button("Remove Today's Record"):
                            if self.attendance_manager.remove_attendance(selected_person):
                                st.success("Removed")
                                st.rerun()
            else:
                st.info("No attendance records found")

        with tab_export:
            st.subheader("Export Data")

            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])

            with col2:
                include_stats = st.checkbox("Include Statistics", value=True)

            if export_format == "CSV":
                # Date range export
                start = st.date_input("Start Date", value=datetime.now().date())
                end = st.date_input("End Date", value=datetime.now().date())

                if st.button("Export CSV"):
                    output = self.attendance_manager.export_csv(
                        start_date=start.strftime('%Y-%m-%d'),
                        end_date=end.strftime('%Y-%m-%d')
                    )
                    st.success(f"Exported to: {output}")

                    with open(output, 'r') as f:
                        st.download_button(
                            label="Download",
                            data=f,
                            file_name=f"attendance_{start}_to_{end}.csv",
                            mime="text/csv"
                        )

            elif export_format == "JSON":
                if st.button("Export JSON"):
                    output = self.attendance_manager.export_json(include_stats=include_stats)
                    st.success(f"Exported to: {output}")

            elif export_format == "Excel":
                if st.button("Export Excel"):
                    output = self.attendance_manager.export_excel()
                    st.success(f"Exported to: {output}")

    def statistics_page(self):
        """View comprehensive statistics."""
        st.subheader("Attendance Statistics")

        # Time period selector
        period = st.radio("Time Period", ["Today", "Last 7 Days", "Last 30 Days", "All Time"])

        days_map = {"Today": 1, "Last 7 Days": 7, "Last 30 Days": 30, "All Time": 365}
        days = days_map[period]

        # Get statistics
        stats = self.attendance_manager.get_statistics(days=days)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Attendance", stats['total_attendance'])

        with col2:
            st.metric("Unique Persons", stats['unique_persons'])

        with col3:
            st.metric("Daily Average", stats['daily_average'])

        with col4:
            count_today = self.attendance_manager.get_attendance_count_today()
            st.metric("Today", count_today)

        st.markdown("---")

        # Charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Daily Attendance")
            if stats['last_7_days']:
                dates = sorted(stats['last_7_days'].keys())
                values = [stats['last_7_days'][d] for d in dates]
                chart_data = pd.DataFrame({"Date": dates, "Count": values})
                st.bar_chart(chart_data.set_index("Date"))
            else:
                st.info("No data available")

        with col_chart2:
            st.subheader("Most Regular Attendees")
            if stats['most_regular']:
                names = [d['name'] for d in stats['most_regular']]
                counts = [d['count'] for d in stats['most_regular']]
                chart_data = pd.DataFrame({"Name": names, "Count": counts})
                st.bar_chart(chart_data.set_index("Name"))
            else:
                st.info("No data available")

        # First arrival today
        if stats['first_arrival']:
            st.markdown("---")
            st.subheader("Today's First Arrival")
            col_first, col_late = st.columns(2)
            with col_first:
                st.write(f"**Name:** {stats['first_arrival']['name']}")
            with col_late:
                st.write(f"**Time:** {stats['first_arrival']['time']}")

    def settings_page(self):
        """Settings page."""
        st.subheader("Detection Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.detection_mode = st.selectbox(
                "Detection Mode",
                ["auto", "strict", "fast"],
                index=["auto", "strict", "fast"].index(st.session_state.detection_mode)
            )
            st.caption("Auto: Balanced | Strict: Fewer false positives | Fast: Speed priority")

        with col2:
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0,
                max_value=100,
                value=int(st.session_state.confidence_threshold),
                help="Lower = stricter recognition"
            )

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.session_state.equalize_histogram = st.checkbox(
                "Equalize Histogram",
                value=st.session_state.equalize_histogram,
                help="Improve contrast for better detection in varying lighting"
            )

        with col4:
            # Lighting info
            if st.button("Test Camera Lighting"):
                if self.camera.start():
                    success, frame = self.camera.get_frame()
                    if success:
                        lighting = self.face_detector.assess_lighting_conditions(frame)
                        st.write(f"**Condition:** {lighting['condition']}")
                        st.write(f"**Mean Brightness:** {lighting['mean_brightness']:.1f}")
                        st.write(f"**Std Deviation:** {lighting['std_deviation']:.1f}")
                        st.write(f"**Recommendation:** {lighting['recommendation']}")
                    self.camera.stop()
                else:
                    st.error("Could not start camera")

        st.markdown("---")

        # Danger zone
        st.subheader("Data Management")
        st.warning("These actions cannot be undone!")

        col_clear_att, col_clear_all = st.columns(2)

        with col_clear_att:
            if st.button("Clear Attendance Data"):
                if st.confirm("Clear all attendance records?"):
                    self.attendance_manager.clear_all()
                    st.success("Cleared attendance data")

        with col_clear_all:
            if st.button("Clear All Data"):
                if st.confirm("Clear ALL data including registered faces?"):
                    self.face_recognizer.clear_all()
                    self.attendance_manager.clear_all()
                    st.success("Cleared all data")

        # About
        st.markdown("---")
        st.caption("Smart Attendance System - Week 4: Computer Vision")
        st.caption("Built with OpenCV and Streamlit")


def main():
    """Main entry point."""
    app = StreamlitAttendanceApp()
    app.run()


if __name__ == '__main__':
    main()