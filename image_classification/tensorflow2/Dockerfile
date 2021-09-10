FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends time \
    ca-certificates \
    build-essential \
    wget unzip

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY . /workspace
WORKDIR /workspace
ENTRYPOINT ["python", "mlcube.py"]