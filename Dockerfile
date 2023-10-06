FROM gramineproject/gramine:latest

# Install Python (adjust the version if needed)
RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /

COPY app /
COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]