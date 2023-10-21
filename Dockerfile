FROM gramineproject/gramine:v1.5

# Update package lists and install essential packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        python3-pip \
        curl \
        libsgx-dcap-default-qpl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --config python3

# Verify Python and pip installation
RUN python3 --version && pip3 --version

WORKDIR /

COPY app /
COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "main.py"]
