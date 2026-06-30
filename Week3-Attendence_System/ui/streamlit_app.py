"""Enhanced Streamlit UI for Smart Attendance System."""

import sys
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
    """Streamlit-based attendance application with dark tech theme."""

    def __init__(self):
        """Initialize the app."""
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()

        # Session state
        if 'detection_mode' not in st.session_state:
            st.session_state.detection_mode = 'auto'
        if 'confidence_threshold' not in st.session_state:
            st.session_state.confidence_threshold = 60
        if 'equalize_histogram' not in st.session_state:
            st.session_state.equalize_histogram = True

    def _inject_dark_theme_css(self):
        """Inject professional dark theme CSS."""
        st.markdown("""
            <style>
            /* Professional Dark Tech Theme */
            .stApp {
                background-color: #0d1117;
            }

            /* Header styling */
            header[data-testid="stHeader"] {
                background-color: #161b22 !important;
            }

            /* Main title */
            h1 {
                color: #58a6ff !important;
                font-family: 'Segoe UI', sans-serif;
                font-weight: 600;
                letter-spacing: -0.5px;
            }

            /* All headings */
            h2, h3, h4 {
                color: #e6edf3 !important;
                font-family: 'Segoe UI', sans-serif;
            }

            /* Sidebar */
            section[data-testid="stSidebar"] {
                background-color: #161b22;
                border-right: 1px solid #30363d;
            }
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] li {
                color: #8b949e !important;
            }

            /* Radio buttons in sidebar */
            div[data-testid="stRadio"] label {
                color: #e6edf3 !important;
                padding: 12px 16px;
                border-radius: 8px;
                margin: 4px 0;
                background-color: #21262d;
                transition: all 0.2s ease;
            }
            div[data-testid="stRadio"] label:hover {
                background-color: #30363d;
            }

            /* Primary buttons - cyan glow */
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 15px;
                transition: all 0.25s ease;
                box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3);
            }
            .stButton > button[kind="primary"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
            }
            .stButton > button[kind="primary"]:disabled {
                background: #21262d;
                color: #484f58;
                box-shadow: none;
            }

            /* Secondary buttons */
            .stButton > button:not([kind="primary"]) {
                background-color: #21262d;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px 20px;
                transition: all 0.2s ease;
            }
            .stButton > button:not([kind="primary"]):hover {
                background-color: #30363d;
                border-color: #484f58;
            }

            /* Input fields */
            .stTextInput > div > div > input {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
            }
            .stTextInput > div > div > input:focus {
                border-color: #0ea5e9;
                box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
            }

            /* Checkboxes */
            .stCheckbox > label {
                color: #8b949e;
            }

            /* Select boxes */
            .stSelectbox > div > div {
                background-color: #0d1117;
                border: 1px solid #30363d;
                color: #e6edf3;
            }

            /* Date inputs */
            .stDateInput > div > div {
                background-color: #0d1117;
                border: 1px solid #30363d;
            }

            /* Metrics */
            div[data-testid="stMetricCard"] {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
            }

            /* Dataframes */
            .stDataFrame {
                background-color: #0d1117;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                color: #8b949e;
                padding: 12px 20px;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                color: #e6edf3;
                border-bottom: 2px solid #0ea5e9;
            }

            /* Success messages */
            .stSuccess {
                background-color: rgba(35, 134, 54, 0.15);
                color: #3fb950;
                border: 1px solid #238636;
                border-radius: 8px;
            }

            /* Error messages */
            .stError {
                background-color: rgba(248, 81, 73, 0.15);
                color: #f85149;
                border: 1px solid #da3633;
                border-radius: 8px;
            }

            /* Info messages */
            .stInfo {
                background-color: rgba(56, 139, 253, 0.15);
                color: #58a6ff;
                border: 1px solid #388bfd;
                border-radius: 8px;
            }

            /* Warning messages */
            .stWarning {
                background-color: rgba(187, 128, 9, 0.15);
                color: #d29922;
                border: 1px solid #9e6a03;
                border-radius: 8px;
            }

            /* Camera input styling */
            .stCameraInput > div > div {
                border: 2px dashed #30363d;
                border-radius: 12px;
                background-color: #161b22;
            }

            /* Divider */
            hr {
                border-color: #30363d;
            }

            /* Download buttons */
            .stDownloadButton > button {
                background-color: #238636;
                color: white;
                border-radius: 8px;
            }
            .stDownloadButton > button:hover {
                background-color: #2ea043;
            }

            /* Expander */
            .streamlit-expanderHeader {
                background-color: #161b22;
                color: #e6edf3;
                border-radius: 8px;
            }

            /* Caption text */
            .stCaption {
                color: #6e7681;
            }
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

        # Apply settings
        self.face_detector.equalize_histogram = st.session_state.equalize_histogram
        self.face_detector.set_detection_mode(st.session_state.detection_mode)
        self.face_recognizer.set_confidence_threshold(st.session_state.confidence_threshold)

        # Apply dark theme
        self._inject_dark_theme_css()

        # Title
        st.title("Smart Attendance System")

        st.markdown("---")

        # Sidebar
        self.sidebar()

        # Main content
        page = st.sidebar.radio("Navigate", [
            "🏠 Home",
            "👤 Register Face",
            "📋 View Attendance",
            "📊 Statistics",
            "⚙️ Settings"
        ])

        if page == "🏠 Home":
            self.home_page()
        elif page == "👤 Register Face":
            self.register_page()
        elif page == "📋 View Attendance":
            self.attendance_page()
        elif page == "📊 Statistics":
            self.statistics_page()
        elif page == "⚙️ Settings":
            self.settings_page()

    def sidebar(self):
        """Render sidebar."""
        st.sidebar.title("Menu")

        # Model info
        with st.sidebar.expander("Model Info", expanded=False):
            info = self.face_recognizer.get_model_info()
            st.write(f"**Total Samples:** {info['total_samples']}")
            st.write(f"**Unique Persons:** {info['unique_persons']}")
            if info['persons']:
                st.write("**Registered:**")
                for p in info['persons']:
                    st.write(f"  • {p}")

        # Today's stats
        st.sidebar.markdown("---")
        count = self.attendance_manager.get_attendance_count_today()
        st.sidebar.metric("Present Today", count)

    def home_page(self):
        """Home page with attendance."""
        st.header("Take Attendance")

        col_camera, col_today = st.columns([3, 2])

        with col_camera:
            # Camera input
            camera_img = st.camera_input(
                "Capture your face",
                help="Position your face in the camera"
            )

            col_attend_btn, col_clear = st.columns([2, 1])
            with col_attend_btn:
                mark_btn = st.button(
                    "✅ Mark Attendance",
                    type="primary",
                    use_container_width=True,
                    disabled=not camera_img
                )

            if mark_btn:
                try:
                    bytes_data = camera_img.getvalue()
                    img = Image.open(io.BytesIO(bytes_data))
                    frame = np.array(img)[:, :, ::-1]

                    faces = self.face_detector.detect_faces(frame)

                    if len(faces) == 0:
                        st.error("❌ No face detected. Please position your face in the camera.")
                    else:
                        results = self.face_recognizer.recognize_faces(frame, faces)

                        if results:
                            for result in results:
                                self.attendance_manager.mark_attendance(result['name'])
                            names = ", ".join(r['name'] for r in results)
                            st.success(f"✓ Marked: {names}")
                        else:
                            st.warning("⚠️ No matching faces found. Please register first.")

                except Exception as e:
                    st.error(f"Error: {e}")

            st.caption("💡 Register faces in the 'Register Face' tab first")

        with col_today:
            st.subheader("Today's Attendance")
            attendance = self.attendance_manager.get_today_attendance()

            if attendance:
                df = pd.DataFrame(attendance)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.markdown(f"**Total: {len(attendance)} present**")
            else:
                st.info("No attendance recorded today")

    def register_page(self):
        """Register new face."""
        st.header("Register New Person")

        # Name input
        col_name, col_multi = st.columns([3, 1])

        with col_name:
            name = st.text_input("Enter Name", key="reg_name_input", placeholder="Type name here...")

        with col_multi:
            st.write("")
            st.write("")
            multiple = st.checkbox("Multiple Samples", value=True, help="Add more samples for better recognition")

        st.markdown("---")

        # Camera and preview
        col_cam, col_prev = st.columns(2)

        with col_cam:
            camera_img = st.camera_input("Take Photo", help="Use camera to capture face")

            reg_btn = st.button(
                "Register",
                type="primary",
                disabled=not camera_img,
                use_container_width=True
            )

            if reg_btn:
                if not name:
                    st.error("Please enter a name")
                else:
                    try:
                        bytes_data = camera_img.getvalue()
                        img = Image.open(io.BytesIO(bytes_data))
                        frame = np.array(img)[:, :, ::-1]

                        faces = self.face_detector.detect_faces(frame)

                        if len(faces) == 0:
                            st.error("❌ No face detected. Please position your face in the camera.")
                        else:
                            face_box = faces[0]
                            success = self.face_recognizer.register_face_box(
                                frame, face_box, name,
                                add_multiple=multiple
                            )

                            if success:
                                count = self.face_recognizer.get_registration_count(name)
                                st.success(f"✓ Registered: {name} ({count} samples)")
                            else:
                                st.error("❌ Failed to register face")

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_prev:
            st.subheader("Registered")

            names = self.face_recognizer.get_registered_names()
            if names:
                for n in names:
                    count = self.face_recognizer.get_registration_count(n)
                    st.write(f"• {n}: {count} samples")
            else:
                st.info("No users registered")

        st.markdown("---")

        # Registered users list with management
        col_list, col_manage = st.columns([3, 1])

        with col_list:
            st.subheader("All Registered Users")
            names = self.face_recognizer.get_registered_names()

            if names:
                data = [{"Name": n, "Samples": self.face_recognizer.get_registration_count(n)} for n in names]
                df = pd.DataFrame(data)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.caption(f"Total: {len(names)} registered")
            else:
                st.info("No users registered")

        with col_manage:
            st.subheader("Manage")
            selected = st.selectbox("Remove person", [""] + names)

            if st.button("Remove", use_container_width=True):
                if selected:
                    if self.face_recognizer.clear_person(selected):
                        st.success(f"Removed: {selected}")
                        st.rerun()
                    else:
                        st.error("Failed to remove")

            if st.button("Clear All", use_container_width=True):
                self.face_recognizer.clear_all()
                st.success("Cleared all faces")
                st.rerun()

    def attendance_page(self):
        """View attendance records."""
        st.header("Attendance Records")

        tab_today, tab_date, tab_person, tab_export = st.tabs(["Today", "By Date", "By Person", "Export"])

        with tab_today:
            attendance = self.attendance_manager.get_today_attendance()

            if attendance:
                df = pd.DataFrame(attendance)
                st.dataframe(df, hide_index=True, use_container_width=True)

                col_count, col_dl = st.columns(2)
                with col_count:
                    st.markdown(f"**Total: {len(attendance)}**")

                with col_dl:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"attendance_{date.today()}.csv",
                        "text/csv",
                        use_container_width=True
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
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download",
                        csv,
                        f"attendance_{date_str}.csv",
                        "text/csv",
                        use_container_width=True
                    )
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
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    stats = self.attendance_manager.get_person_statistics(selected_person)
                    st.markdown(f"**Total Days:** {stats['total_days']}")
                    st.markdown(f"**First Seen:** {stats['first_seen']}")
                    st.markdown(f"**Last Seen:** {stats['last_seen']}")
                    st.markdown(f"**Streak:** {stats['streak']} days")

                    if stats['last_seen'] == date.today().isoformat():
                        if st.button("Remove Today's Record"):
                            if self.attendance_manager.remove_attendance(selected_person):
                                st.success("Removed")
                                st.rerun()
            else:
                st.info("No records found")

        with tab_export:
            st.subheader("Export Data")

            col_fmt, col_stats = st.columns(2)
            with col_fmt:
                export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
            with col_stats:
                include_stats = st.checkbox("Include Statistics", value=True)

            if export_format == "CSV":
                start = st.date_input("Start Date", value=datetime.now().date())
                end = st.date_input("End Date", value=datetime.now().date())

                if st.button("Export CSV"):
                    output = self.attendance_manager.export_csv(
                        start_date=start.strftime('%Y-%m-%d'),
                        end_date=end.strftime('%Y-%m-%d')
                    )
                    with open(output, 'r') as f:
                        st.download_button(
                            "Download CSV",
                            f.read(),
                            f"attendance_{start}_{end}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    st.success("✓ Exported!")

            elif export_format == "JSON":
                if st.button("Export JSON"):
                    output = self.attendance_manager.export_json(include_stats=include_stats)
                    with open(output, 'r') as f:
                        st.download_button(
                            "Download JSON",
                            f.read(),
                            "attendance.json",
                            "application/json",
                            use_container_width=True
                        )
                    st.success("✓ Exported!")

            elif export_format == "Excel":
                if st.button("Export Excel"):
                    output = self.attendance_manager.export_excel()
                    with open(output, 'rb') as f:
                        st.download_button(
                            "Download Excel",
                            f.read(),
                            "attendance.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    st.success("✓ Exported!")

    def statistics_page(self):
        """View statistics."""
        st.header("Attendance Statistics")

        period = st.radio("Time Period", ["Today", "Last 7 Days", "Last 30 Days", "All Time"])

        days_map = {"Today": 1, "Last 7 Days": 7, "Last 30 Days": 30, "All Time": 365}
        days = days_map[period]

        stats = self.attendance_manager.get_statistics(days=days)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total", stats['total_attendance'])
        with col2:
            st.metric("Persons", stats['unique_persons'])
        with col3:
            st.metric("Daily Avg", stats['daily_average'])
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
                st.info("No data")

        with col_chart2:
            st.subheader("Top Attendees")
            if stats['most_regular']:
                names = [d['name'] for d in stats['most_regular']]
                counts = [d['count'] for d in stats['most_regular']]
                chart_data = pd.DataFrame({"Name": names, "Count": counts})
                st.bar_chart(chart_data.set_index("Name"))
            else:
                st.info("No data")

        if stats['first_arrival']:
            st.markdown("---")
            st.subheader("First Arrival Today")
            col_first, col_time = st.columns(2)
            with col_first:
                st.markdown(f"**Name:** {stats['first_arrival']['name']}")
            with col_time:
                st.markdown(f"**Time:** {stats['first_arrival']['time']}")

    def settings_page(self):
        """Settings page."""
        st.header("Settings")

        # Detection settings
        st.subheader("Detection")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.detection_mode = st.selectbox(
                "Detection Mode",
                ["auto", "strict", "fast"],
                index=["auto", "strict", "fast"].index(st.session_state.detection_mode)
            )
            st.caption("Auto: Balanced | Strict: Accurate | Fast: Quick")

        with col2:
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                0, 100, int(st.session_state.confidence_threshold),
                help="Lower = stricter"
            )

        col3, col4 = st.columns(2)

        with col3:
            st.session_state.equalize_histogram = st.checkbox(
                "Equalize Histogram",
                value=st.session_state.equalize_histogram,
                help="Improve contrast"
            )

        st.markdown("---")

        # Data management
        st.subheader("Data Management")

        col_clear_att, col_clear_all = st.columns(2)

        with col_clear_att:
            if st.button("Clear Attendance Data", use_container_width=True):
                self.attendance_manager.clear_all()
                st.success("Cleared attendance data")

        with col_clear_all:
            if st.button("Clear All Data", use_container_width=True):
                self.face_recognizer.clear_all()
                self.attendance_manager.clear_all()
                st.success("Cleared all data")

        st.markdown("---")
        st.caption("Smart Attendance System - Week 4: Computer Vision")


def main():
    """Main entry point."""
    app = StreamlitAttendanceApp()
    app.run()


if __name__ == '__main__':
    main()