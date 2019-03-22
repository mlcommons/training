ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.03-py3
FROM ${FROM_IMAGE_NAME}

# Install Python dependencies
WORKDIR /workspace/recommendation

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY negative_sampling_cpp ./negative_sampling_cpp
WORKDIR /workspace/recommendation/negative_sampling_cpp
RUN python setup.py build --debug
RUN python setup.py install

# Copy NCF code and build
WORKDIR /workspace/recommendation
COPY . .
