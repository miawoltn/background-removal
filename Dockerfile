# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN adduser --disabled-password --gecos '' appuser
WORKDIR /home/appuser

RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]