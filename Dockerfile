FROM python:3.8.6-buster

WORKDIR /deps

RUN wget -q --show-progress http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip && \
    unzip -P iagreetotheeula SC2.3.16.1.zip && \
    rm SC2.3.16.1.zip
RUN wget -q --show-progress https://github.com/oxwhirl/smac/releases/download/v1/SMAC_Maps_V1.tar.gz && \
    tar -xf SMAC_Maps_V1.tar.gz && \
    rm SMAC_Maps_V1.tar.gz && \
    mv SMAC_Maps StarCraftII/Maps/

ENV SC2PATH /deps/StarCraftII

RUN apt update && apt install -y swig
RUN python3 -m venv virtual
RUN git clone https://github.com/oxwhirl/smac && \
    cd smac && \
    git checkout d41d303 && \
    /deps/virtual/bin/python3 -m pip install -e .
RUN virtual/bin/python3 -m pip install torch
RUN virtual/bin/python3 -m pip install matplotlib

WORKDIR /app
