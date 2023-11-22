FROM python:3.11
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV PYTHON_VERSION 3.11.5
RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN apt-get install -y python3.11 python3-pip python3-dev build-essential python3-venv
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD python3 src/ui.py