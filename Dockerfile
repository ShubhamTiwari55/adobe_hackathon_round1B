# Dockerfile

# Use a specific Python version for reproducibility
FROM --platform=linux/amd64 python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by Tesseract and OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire project into the container
# This includes run.py, the 'models' folder, and your lgbm_relevance.joblib
COPY . .

# Define the command to run your script when the container starts
# This will process /app/input and write to /app/output
CMD ["python", "run.py"]