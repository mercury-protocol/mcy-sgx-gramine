FROM ubuntu:20.04

# Configuration
ENV SGX_SDK_BIN=sgx_linux_x64_sdk_2.19.100.3.bin
ENV DISTRO=ubuntu20.04-server

# Install dependencies
RUN set -xe - y && \
    apt-get update -y && \
    apt-get install -y python3-pip && \
    apt-get install -y build-essential curl wget && \
    apt-get clean

# Install SXX SDK
RUN wget https://download.01.org/intel-sgx/latest/linux-latest/distro/${DISTRO}/${SGX_SDK_BIN} && \
    chmod +x ${SGX_SDK_BIN} && \
    echo -e 'no\n/opt' | ./${SGX_SDK_BIN} && \
    rm -rf ${SGX_SDK_BIN}
ENV PATH="/opt/sgxsdk/bin:${PATH}"

# Install python packages
COPY requirements.txt requirements_test.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory and copy source code
WORKDIR /app
COPY app ./
