FROM python:3.8.6-buster

WORKDIR /root

ENV SC2PATH /app/StarCraftII
RUN apt update && apt install -y swig
RUN python3 -m venv virtual
RUN git clone https://github.com/oxwhirl/smac && cd smac && git checkout d41d303 && /root/virtual/bin/python3 -m pip install -e .
RUN virtual/bin/python3 -m pip install torch
RUN virtual/bin/python3 -m pip install matplotlib

WORKDIR /app
