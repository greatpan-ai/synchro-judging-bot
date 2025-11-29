# Start from a base Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (main.py, as_judging.json, static/ directory)
COPY . /app

# Expose the port (FastAPI default)
EXPOSE 8000

# Define the command to run the application using Uvicorn
# 'main:app' refers to the 'app' object in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]