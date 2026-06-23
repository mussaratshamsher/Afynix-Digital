FROM python:3.10-slim

WORKDIR /workspace

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set PYTHONPATH to search for src directory inside the project root
ENV PYTHONPATH=/workspace

# Expose port 7860 (Hugging Face Spaces requirement)
EXPOSE 7860

# Start Uvicorn to run the FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
