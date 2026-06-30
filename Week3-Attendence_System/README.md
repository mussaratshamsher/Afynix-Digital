---
title: Smart Attendance System
emoji: 📸
colorFrom: green
colorTo: blue
sdk: streamlit
python_version: 3.11
app_file: ui/streamlit_app.py
pinned: false
---

# Smart Attendance System 📸

A **face recognition-based attendance system** that uses your webcam to detect and recognize faces, automatically marking attendance with timestamps. Perfect for classrooms, offices, or any environment where you need to track attendance digitally.

![Demo](https://img.shields.io/badge/Build-Passing-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red)

---

## Features

### Core Functionality
- **Real-time Face Detection** using Haar Cascade Classifier
- **Face Recognition** with histogram comparison and feature matching
- **Automatic Attendance Marking** with timestamps
- **Duplicate Prevention** (one entry per day per person)
- **Export Options**: CSV, JSON, Excel

### Detection
- Adaptive lighting compensation with CLAHE
- Multi-scale detection for better accuracy
- Configurable detection modes (Auto/Strict/Fast)
- Quality assessment for detected faces

### Recognition
- Multiple samples per person for robust recognition
- Confidence scoring (0-100%)
- Configurable confidence threshold
- Person management (add/remove/clear)

### Dashboard
- Live camera input via browser
- Registration with camera capture
- View attendance records
- Statistics and analytics
- Date range filtering

---

## Usage

### Home Page
1. Take a photo using your browser camera
2. Click "Mark Attendance"
3. System detects and recognizes your face
4. Attendance is automatically recorded

### Register New Person
1. Go to "Register Face" page
2. Enter your name
3. Take a photo with your camera
4. Click "Register"

### View Attendance
1. Go to "View Attendance" page
2. Filter by date or person
3. Export to CSV/JSON/Excel

### Statistics
1. Go to "Statistics" page
2. View attendance analytics
3. See daily trends and regular attendees

---

## Installation

### Local Deployment

```bash
# Clone the repository
git clone https://github.com/yourusername/Afynix-Digital.git
cd Week3-Attendence_System

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ui/streamlit_app.py
```

### Hugging Face Deployment

1. **Create a new Space** on Hugging Face
2. Select **Streamlit** as the SDK
3. Push this code to your GitHub repository
4. Hugging Face will automatically build and deploy

### Docker Deployment

```bash
# Build the image
docker build -t attendance-system .

# Run the container
docker run -p 7860:7860 attendance-system
```

---

## Project Structure

```
Week3-Attendence_System/
├── app.py                    # Main CLI (desktop only)
├── ui/
│   └── streamlit_app.py       # Streamlit web interface
├── src/
│   ├── camera.py            # Webcam capture
│   ├── face detector.py     # Face detection
│   ├── face_recognizer.py  # Face recognition
│   └── attendance_manager.py # Attendance tracking
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── face_data.pkl
├── data/
│   ├── faces/             # Registered face images
│   └── attendance/        # Attendance CSV
├── requirements.txt       # Python dependencies
├── Dockerfile           # Docker configuration
└── README.md           # This file
```

---

## Configuration

### Detection Settings

| Setting | Description | Options |
|---------|------------|---------|
| Detection Mode | Speed/accuracy balance | Auto, Strict, Fast |
| Confidence Threshold | Recognition strictness | 0-100 |
| Equalize Histogram | Improve contrast | On/Off |

---

## Technology Stack

- **Python 3.11+** - Programming language
- **OpenCV** - Computer vision
- **Streamlit** - Web framework
- **Pillow** - Image processing
- **Pandas** - Data handling

---

## License

MIT License - Open Source

---

## Author

**Afynix Digital** - Agentic Hackathon 2025

Week 4: Computer Vision - Face Recognition Attendance System

---

## Acknowledgments

- OpenCV community
- Streamlit team
- All contributors and testers