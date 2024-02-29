FROM ubuntu:20.04


COPY requirements.txt .
COPY . .
RUN apt update
RUN apt install python3 python3-pip -y
RUN pip install -r requirements.txt


