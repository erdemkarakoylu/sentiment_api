# === BASE Stage === #
# This BASE stage creates the base image for CUDA & PyTorch application.
FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04 AS base
RUN apt update
RUN apt-get -y install vim

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

WORKDIR /app
# install pytorch with support for a specific CUDA version
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# === BUILD Stage ===
FROM base AS build

COPY utils/stash_big_models.py .
RUN python3 stash_big_models.py

# === PRODUCTION Stage ===
FROM build AS production
COPY app.py /app/


