# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (including mlruns) into the container
COPY src/ /app/src/

COPY mlruns/ /app/mlruns/

# Expose port 8000
EXPOSE 8000

# Command to run the app
# We also need to tell MLflow where to find the tracking data inside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]