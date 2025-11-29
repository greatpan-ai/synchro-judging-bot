# Start from a base Python image
FROM python:3.11-slim

# Install necessary system dependencies for OpenCV (libGL, libXext, etc.)
# The 'apt-get update' must be run first.
RUN apt-get update && \ 
    apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (main.py, as_judging.json, static/ directory, etc.)
COPY . /app

# Expose the port (FastAPI default)
EXPOSE 8000

# Define the command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]