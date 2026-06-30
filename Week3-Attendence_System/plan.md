# Smart Attendance System (Computer Vision with OpenCV)

> A Smart Attendance System that automatically detects and recognizes faces using a laptop webcam and marks attendance in a CSV file with timestamps.

> **Developer:** Afynix Digital

> **Project:** Agentic Hackathon 2025 - Week 3: Computer Vision

---

## Quick Start

### Installation

```bash
cd Week3-Computer_Vision
pip install -r requirements.txt
```

### Register a New Person

```bash
python app.py --register "YourName"
```

### List Registered Users

```bash
python app.py --list
```

### Run Attendance System

```bash
python app.py
```

### Run Streamlit Dashboard

```bash
streamlit run ui/streamlit_app.py
```

---

## Project Overview

Build a Smart Attendance System that automatically detects and recognizes faces using a laptop webcam and marks attendance in a CSV file with timestamps.

This project will use completely free and open-source tools.

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python 3.11+ |
| Computer Vision | OpenCV |
| Face Detection | Haar Cascade Classifier |
| Face Recognition | face_recognition (dlib-based) |
| Data Handling | Pandas |
| UI (Optional) | Streamlit |
| Attendance Storage | CSV Files |
| Camera Input | Laptop Webcam |
| Environment | Virtual Environment (.venv) |

---

# Project Goals

### Core Features

- Capture live video from laptop webcam
- Detect faces in real-time
- Recognize registered users
- Mark attendance automatically
- Prevent duplicate attendance entries
- Save timestamp with attendance
- Export attendance records to CSV

### Bonus Features

- Streamlit UI
- Attendance dashboard
- Daily attendance reports

---

# Project Structure

```text
smart-attendance-system/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── faces/
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   └── ...
│   │
│   └── attendance/
│       └── attendance.csv
│
├── models/
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── camera.py
│   ├── face_detector.py
│   ├── face_recognizer.py
│   ├── attendance_manager.py
│   └── utils.py
│
├── ui/
│   └── streamlit_app.py
│
└── tests/
    └── test_attendance.py
```

---

# Development Phases

## Phase 1 — Environment Setup

### Tasks

- Create project folder
- Create virtual environment
- Install required libraries
- Download Haar Cascade model
- Verify webcam access

### Deliverables

- Running webcam feed
- Working Python environment

### Test

- Webcam opens successfully
- Frames display without errors

---

## Phase 2 — Face Detection

### Objective

Detect faces from webcam stream.

### Tasks

- Load Haar Cascade classifier
- Capture video frames
- Convert frames to grayscale
- Detect faces
- Draw bounding boxes

### Deliverables

- Real-time face detection

### Test

- Detect single face
- Detect multiple faces
- Test at different distances

---

## Phase 3 — Face Registration

### Objective

Register known users.

### Tasks

- Create faces folder
- Capture or upload face images
- Store images with person name
- Generate face encodings

### Deliverables

- Registered face dataset
- Stored encodings

### Test

- Encoding generated successfully
- Multiple users registered

---

## Phase 4 — Face Recognition

### Objective

Recognize known individuals.

### Tasks

- Load saved encodings
- Compare live face against known faces
- Display recognized name

### Deliverables

- Real-time face recognition

### Test

- Recognize registered users
- Handle unknown users correctly

---

## Phase 5 — Attendance Management

### Objective

Record attendance automatically.

### Tasks

- Create attendance CSV
- Store:
  - Name
  - Date
  - Time
- Prevent duplicate entries
- Mark attendance only once per session

### CSV Format

```csv
Name,Date,Time
John,2026-08-01,09:10:22
Ali,2026-08-01,09:15:30
```

### Deliverables

- Automated attendance logging

### Test

- Duplicate attendance prevented
- Timestamp stored correctly

---

## Phase 6 — Streamlit UI (Bonus)

### Objective

Provide a simple user interface.

### Features

#### Home Screen

- Start Attendance Button
- Stop Attendance Button

#### Attendance View

- Show today's attendance
- Download CSV

### Deliverables

- Functional Streamlit dashboard

### Test

- Buttons work correctly
- CSV download works

---

## Phase 7 — Testing & Optimization

### Functional Testing

#### Face Detection

- Bright lighting
- Dim lighting
- Multiple faces

#### Face Recognition

- Registered users
- Unknown users

#### Attendance

- Single attendance entry
- Multiple users
- CSV generation

### Performance Testing

- Webcam FPS
- Recognition speed
- Memory usage

### Edge Cases

- No face visible
- Multiple people entering together
- Poor lighting conditions
- Webcam disconnects unexpectedly

### Deliverables

- Stable application
- Bug fixes completed

---

# Free Resources

## OpenCV

- Face detection
- Webcam integration

## Haar Cascade

- Pre-trained face detector
- Completely free

## face_recognition

- Free face encoding and recognition

## Pandas

- Attendance CSV management

## Streamlit

- Free local web interface

---

# Final Workflow

```text
Start Application
        │
        ▼
Open Laptop Webcam
        │
        ▼
Detect Faces
        │
        ▼
Recognize User
        │
        ▼
Check Attendance Log
        │
        ├── Already Marked
        │       │
        │       ▼
        │   Ignore Entry
        │
        └── New User
                │
                ▼
        Save Name + Timestamp
                │
                ▼
        Update CSV
                │
                ▼
         Display Success
```

---

# Expected Outcome

A fully functional Smart Attendance System that:

- Uses laptop webcam for real-time monitoring
- Detects faces using Haar Cascade
- Recognizes registered users
- Records attendance automatically
- Prevents duplicate entries
- Stores attendance with timestamps
- Exports attendance records to CSV
- Optionally provides a simple Streamlit dashboard
