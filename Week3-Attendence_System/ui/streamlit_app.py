"""Enhanced Streamlit UI for Smart Attendance System."""

import streamlit as st
import numpy as np
import io
from pathlib import Path
from datetime import datetime, date
import pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.attendance_manager import AttendanceManager


class StreamlitAttendanceApp:
    """Streamlit-based attendance application with enhanced features."""

    def __init__(self):
        """Initialize the app."""
        # Initialize components (no Camera - use browser input)
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()

        # Session state for settings
        if 'detection_mode' not in st.session_state:
            st.session_state.detection_mode = 'auto'
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 60  # 60% for easier attendance recognition
        if 'equalize_histogram' not in st.session_state:
            st.session_state.equalize_histogram = True
        if 'attendance_running' not in st.session_state:
            st.session_state.attendance_running = False
        if 'registration_name' not in st.session_state:
            st.session_state.registration_name = ''
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'

    def _inject_theme_css(self):
        """Inject custom theme CSS with animations."""
        is_dark = st.session_state.theme == 'dark'
        bg = "#0E1117" if is_dark else "#FFFFFF"
        sidebar_bg = "#262730" if is_dark else "#F0F2F6"
        text = "#FAFAFA" if is_dark else "#31333F"
        primary = "#00E676"
        secondary = "#31333F" if is_dark else "#F0F2F6"
        header_bg = "#1A1A1A" if is_dark else "#FFFFFF"
        header_text = "#FFFFFF" if is_dark else "#31333F"

        st.markdown(f"""
            <style>
            /* Main app background with animation */
            .stApp {{
                background-color: {bg};
                animation: fadeIn 0.5s ease-in;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            @keyframes slideIn {{
                from {{ transform: translateY(-20px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            /* Header/Title area */
            header[data-testid="stHeader"] {{
                background-color: {header_bg} !important;
            }}
            header[data-testid="stHeader"] * {{
                color: {header_text} !important;
            }}
            /* Main title styling */
            h1 {{
                color: {text} !important;
                animation: slideIn 0.5s ease-out;
            }}
            /* Text colors */
            .stMarkdown, .stText, p, span, li, div[data-testid="stMetricValue"] {{
                color: {text} !important;
            }}
            /* Headings with animation */
            h2, h3, h4 {{
                color: {text} !important;
                animation: slideIn 0.4s ease-out;
            }}
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: {sidebar_bg};
            }}
            section[data-testid="stSidebar"] * {{
                color: {text} !important;
            }}
            /* Buttons with hover animation */
            .stButton > button {{
                background-color: {primary};
                color: #000000;
                border: none;
                transition: all 0.3s ease;
                animation: slideIn 0.3s ease-out;
            }}
            .stButton > button:hover {{
                background-color: #00C853;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            /* Cards with animation */
            div[data-testid="stMetricCard"] {{
                background-color: {sidebar_bg};
                color: {text};
                border-radius: 10px;
                animation: slideIn 0.4s ease-out;
            }}
            /* Input fields */
            .stTextInput > div > div > input {{
                background-color: {secondary};
                color: {text};
                border-radius: 5px;
            }}
            /* Dataframes */
            .stDataFrame {{
                background-color: {bg};
            }}
            /* Success/Error messages with animation */
            .stSuccess, .stError {{
                animation: slideIn 0.3s ease-out, pulse 0.3s ease-out;
            }}
            /* Video placeholder with border */
            div[data-testid="stImage"] {{
                border-radius: 10px;
                overflow: hidden;
            }}
            /* Radio buttons */
            div[data-testid="stRadio"] > label {{
                padding: 5px 10px;
                border-radius: 5px;
                transition: all 0.2s ease;
            }}
            div[data-testid="stRadio"] > label:hover {{
                background-color: {secondary};
            }}
            /* Tabs with animation */
            .stTabs [data-baseweb="tab"] {{
                transition: all 0.2s ease;
            }}
            .stTabs [data-baseweb="tab"]:hover {{
                background-color: {secondary};
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
        """Home page with camera input for attendance."""
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Take Attendance")

            # Camera input
            camera_img = st.camera_input("Take photo for attendance", help="Capture face for attendance")

            if st.button("Mark Attendance", type="primary", disabled=not camera_img):
                try:
                    bytes_data = camera_img.getvalue()
                    img = Image.open(io.BytesIO(bytes_data))
                    frame = np.array(img)[:, :, ::-1]  # RGB to BGR

                    # Detect and recognize
                    faces = self.face_detector.detect_faces(frame)

                    if len(faces) == 0:
                        st.error("No face detected")
                    else:
                        results = self.face_recognizer.recognize_faces(frame, faces)

                        if results:
                            for result in results:
                                self.attendance_manager.mark_attendance(result['name'])
                            names = ", ".join(r['name'] for r in results)
                            st.success(f"Marked: {names}")
                        else:
                            st.warning("No matching faces found")

                except Exception as e:
                    st.error(f"Error: {e}")

            st.markdown("---")
            st.info("Tip: Register faces first in the Register page")

        with col2:
            st.subheader("Today's Attendance")
            attendance = self.attendance_manager.get_today_attendance()

            if attendance:
                df = pd.DataFrame(attendance)
                st.dataframe(df, hide_index=True, height=200)
                st.write(f"**Total Present:** {len(attendance)}")
            else:
                st.info("No attendance recorded today")

    def register_page(self):
        """Register a new face using browser camera input."""
        st.subheader("Register New Person")

        col1, col2 = st.columns([2, 1])

        with col1:
            name = st.text_input("Enter Name", key="reg_name_input")

        with col2:
            st.write("")
            st.write("")
            multiple = st.checkbox("Add Multiple Samples", value=True,
                            help="Add more samples for better recognition")

        st.markdown("---")

        # Use browser camera input (works on Streamlit Cloud)
        col_upload, col_preview = st.columns(2)

        with col_upload:
            st.subheader("Take Photo")
            camera_img = st.camera_input("Take a photo", help="Use your camera to capture face")

            if st.button("Register", type="primary", disabled=not camera_img):
                if not name:
                    st.error("Please enter a name")
                else:
                    try:
                        # Read image from camera input
                        bytes_data = camera_img.getvalue()
                        img = Image.open(io.BytesIO(bytes_data))
                        frame = np.array(img)[:, :, ::-1]  # RGB to BGR

                        # Detect faces
                        faces = self.face_detector.detect_faces(frame)

                        if len(faces) == 0:
                            st.error("No face detected. Please position your face in the camera.")
                        else:
                            face_box = faces[0]
                            # Register
                            success = self.face_recognizer.register_face_box(
                                frame, face_box, name,
                                add_multiple=multiple
                            )

                            if success:
                                count = self.face_recognizer.get_registration_count(name)
                                st.success(f"Registered: {name} ({count} samples)")
                            else:
                                st.error("Failed to register face")

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_preview:
            st.subheader("Registered Faces")
            names = self.face_recognizer.get_registered_names()
            if names:
                for n in names:
                    count = self.face_recognizer.get_registration_count(n)
                    st.write(f"- {n}: {count} samples")
            else:
                st.info("No users registered")

        with col_preview:
            st.subheader("Preview")
            # Show registered face
            face_path = Path(__file__).parent.parent / "data" / "faces" / f"{name}.jpg"
            if face_path.exists():
                face_img = Image.open(face_path)
                # Handle grayscale (1-channel) images
                if face_img.mode == 'L':
                    face_img = face_img.convert('RGB')
                face_img = np.array(face_img)
                st.image(face_img, caption=f"{name}", width='stretch')
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
                try:
                    if self.camera.start():
                        success, frame = self.camera.get_frame()
                        if success and frame is not None:
                            lighting = self.face_detector.assess_lighting_conditions(frame)
                            st.write(f"**Condition:** {lighting['condition']}")
                            st.write(f"**Mean Brightness:** {lighting['mean_brightness']:.1f}")
                            st.write(f"**Std Deviation:** {lighting['std_deviation']:.1f}")
                            st.write(f"**Recommendation:** {lighting['recommendation']}")
                        else:
                            st.error("Could not read frame")
                    else:
                        st.error("Could not start camera")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    try:
                        self.camera.stop()
                    except:
                        pass

        st.markdown("---")

        # Danger zone
        st.subheader("Data Management")

        col_clear_att, col_clear_all = st.columns(2)

        with col_clear_att:
            if st.button("Clear Attendance Data"):
                self.attendance_manager.clear_all()
                st.success("Cleared attendance data")

        with col_clear_all:
            if st.button("Clear All Data"):
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