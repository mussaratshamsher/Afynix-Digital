# Smart Attendance System

A Smart Attendance System that automatically detects and recognizes faces using a laptop webcam and marks attendance in a CSV file with timestamps.

---

## Features

### Core Features
- Real-time face detection using **Haar Cascade Classifier** with adaptive lighting compensation
- Face recognition using histogram comparison and feature matching
- Automatic attendance marking with timestamps
- Duplicate attendance prevention (one entry per day per person)
- CSV export and import

### Enhanced Detection
- **Adaptive lighting compensation** using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Multi-scale detection** for better accuracy
- **Detection modes**: Auto (balanced), Strict (fewer false positives), Fast (speed priority)
- **Lighting condition assessment** with recommendations for testing

### Enhanced Recognition
- **Multiple samples per person** for more robust recognition (base + augmented samples)
- **Confidence scoring** (0-100%) with color-coded visual feedback
- **Alternative match suggestions** when recognition confidence is uncertain
- **Configurable confidence threshold** for different security levels
- **Person management**: Add, remove, clear registered users

### Enhanced Attendance Management
- **Statistics dashboard**: Total attendance, daily average, unique persons, streak tracking
- **Person-specific history**: Individual attendance patterns and statistics
- **Export options**: CSV, JSON, Excel formats
- **Date range filtering**: Filter and export specific date ranges
- **Import capability**: Import attendance from external CSV files
- **Report generation**: Generate pivot tables by date/person

---

## Requirements

### Python Version

- Python 3.11+

### Dependencies

```
opencv-python>=4.8.0
pandas>=2.0.0
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
```

---

## Installation

### 1. Navigate to Project

```bash
cd Week3-Computer_Vision
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Command Line Interface

#### Register a New Person

```bash
python app.py --register "YourName"
```

This will:
- Open your webcam
- Detect your face
- Register you in the system (with multiple samples for better recognition)

#### List Registered Users

```bash
python app.py --list
```

#### Run Attendance System

```bash
python app.py
```

Then:
- Press `q` to quit
- Press `s` to toggle video display

---

### Streamlit Dashboard

#### Run the Dashboard

```bash
streamlit run ui/streamlit_app.py
```

#### Dashboard Pages

1. **Home**: Live camera feed with real-time face detection and recognition
2. **Register Face**: Register new people with camera
3. **View Attendance**: View and export attendance records
4. **Statistics**: View comprehensive attendance statistics and charts
5. **Settings**: Configure detection parameters and manage data

---

## Testing in Different Lighting Conditions

### Using Settings Page

1. Navigate to **Settings** in the dashboard
2. Click **Test Camera Lighting**
3. Check the condition assessment:
   - **Good**: Optimal lighting
   - **Fair**: Acceptable lighting
   - **Poor**: Improve lighting

### Manual Testing Tips

- Ensure face is clearly visible
- Position face centered in frame
- Avoid backlighting (window behind you)
- Use even lighting (not too bright, not too dark)
- Avoid strong shadows on face

---

## Project Structure

```
Week3-Attendence_System/
├── app.py                    # Main CLI application
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── models/
│   ├── haarcascade_frontalface_default.xml   # Haar Cascade model
│   └── face_data.pkl       # Face recognition model
│
├── src/
│   ├── camera.py           # Webcam capture
│   ├── face_detector.py    # Haar Cascade detection with enhancements
│   ├── face_recognizer.py # Face recognition with enhancements
│   ├── attendance_manager.py # Attendance management with stats
│   └── utils.py           # Helper functions
│
├── ui/
│   └── streamlit_app.py   # Enhanced Streamlit dashboard
│
├── data/
│   ├── faces/            # Registered face images
│   └── attendance/       # Attendance CSV files
│
└── tests/
    └── test_attendance.py # Unit tests
```

---

## Attendance CSV Format

```csv
Name,Date,Time
John,2026-06-27,09:10:22
Alice,2026-06-27,09:15:30
```

---

## Configuration

### Detection Settings (Settings Page)

| Setting | Description | Options |
|---------|------------|---------|
| Detection Mode | Balance between speed and accuracy | auto, strict, fast |
| Confidence Threshold | Recognition strictness (lower = stricter) | 0-100 |
| Equalize Histogram | Improve contrast for varying lighting | on/off |

### Detection Mode Comparison

| Mode | Scale Factor | Min Neighbors | Best For |
|------|------------|-------------|----------|
| auto | 1.05 | 3 | Default use |
| strict | 1.20 | 5 | High security |
| fast | 1.15 | 4 | Real-time processing |

---

## Troubleshooting

### Camera Not Opening

- Check if webcam is connected
- Try a different camera index: `python app.py --camera 1`

### Face Not Detected

- Ensure good lighting
- Face should be clearly visible
- Position face centered in frame

### Recognition Issues

- Register multiple samples of the same person
- Lower the confidence threshold in Settings
- Test lighting conditions

### CSV Export Issues

- Ensure pandas is installed: `pip install pandas`

---

## License

Open Source - Free to use

---

## Author

**Afynix Digital** - Agentic Hackathon 2025 - Week 4: Computer Vision

---

## Acknowledgments

- OpenCV for computer vision capabilities
- Streamlit for the web dashboard
- All contributors and testers