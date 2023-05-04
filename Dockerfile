# Base image
FROM python:3.9-slim-buster

RUN apt-get update && apt-get install libgl1 libglib2.0-0 gcc clang clang-tools cmake -y

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host", "0.0.0.0"]

# production
# CMD [ "waitress-serve", "--call" , "app:create_app"]
