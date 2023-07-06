# Base image
FROM python:3.9-slim-buster

RUN apt-get update && apt-get install libgl1 libglib2.0-0 gcc clang clang-tools cmake -y

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
COPY requirements-no-deps.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-deps -r requirements-no-deps.txt

# Copy the application code into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
