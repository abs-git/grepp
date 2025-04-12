FROM nvcr.io/nvidia/tensorrt:24.05-py3 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Seoul
ENV ROOT=/workspace
ENV PYTHONPATH=$ROOT

WORKDIR $ROOT

RUN apt-get update && \
    apt-get -y install --no-install-recommends openssh-client cmake vim \
    wget curl git iputils-ping net-tools htop build-essential && \
    /opt/tensorrt/python/python_setup.sh

RUN python -m pip install --upgrade pip

FROM base AS stage1

RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install torchsummary==1.5.1

RUN pip3 install --no-cache-dir scikit-learn==1.6.1 numpy==1.26.4 \
    onnx==1.17.0 onnxruntime==1.20.1 \
    opencv-python==4.10.0.84

RUN pip3 install easydict matplotlib

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libglib2.0-0

FROM stage1 AS app
