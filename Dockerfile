# Base image
FROM python:3.9-slim-buster

RUN adduser --disabled-password --gecos '' appuser

# Set the working directory
WORKDIR /home/appuser


# install os dependencies
RUN apt-get update && apt-get install libgl1 libglib2.0-0 gcc clang clang-tools cmake tk -y

# Copy the requirements file into the container
COPY requirements.txt .
COPY requirements-no-deps.txt .

# Install dependencies
# RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-deps -r requirements-no-deps.txt
ENV LD_PRELOAD=/usr/local/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0

# Copy the application code into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

CMD ["gunicorn", "--workers=4", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
