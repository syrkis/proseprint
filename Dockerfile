FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && \
    apt-get install -y \
    python3.10 python3.10-distutils python3.10-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py --force-reinstall && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3.10 -m pip install -r requirements.txt

RUN poetry 3.10 -m pip install torch==1.10.1+cu111 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
