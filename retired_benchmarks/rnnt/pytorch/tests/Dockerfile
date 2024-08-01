FROM nvcr.io/nvidia/pytorch:20.10-py3

COPY tests/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY requirements.txt .
RUN pip install -r requirements.txt



WORKDIR /code

CMD bash
