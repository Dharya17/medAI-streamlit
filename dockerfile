# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and ensure the app runs in a non-interactive mode
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY . .

# Install the dependencies specified in the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 8000

# Run Streamlit when the container starts
CMD ["streamlit", "run", "MedAI.py"]
