# Smart Attendance System - Dockerfile
# For Hugging Face Spaces deployment

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directories
RUN mkdir -p data/faces data/attendance models

# Expose port for Hugging Face
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]