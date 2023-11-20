FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get -y update && apt-get install -y --no-install-recommends\  
 python3-pip\
 git\
 libgl1-mesa-glx\
 libglib2.0-0 \
 curl 

WORKDIR /opt/program
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/xinntao/Real-ESRGAN.git
# COPY scripts/__init__.py Real-ESRGAN/realesrgan
# RUN pip3 install -r Real-ESRGAN/requirements.txt
###
# RUN pip install awscli
# # RUN mkdir /opt/ml /opt/ml/model
# RUN aws s3 cp s3://sagemaker-us-east-2-169385451286/output/gan-2023-11-13-18-52-22-074/output/model.tar.gz /opt/ml/model
# WORKDIR /opt/ml/model
# RUN tar -xvf model.tar.gz 
# WORKDIR /opt/program
###

ENV PATH="/opt/program/Real-ESRGAN:${PATH}"
COPY utils.py Real-ESRGAN/realesrgan
COPY main.py Real-ESRGAN/.
COPY predictor.py Real-ESRGAN/.
COPY serve Real-ESRGAN/.

RUN apt-get install -y gpg sed
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    apt-get update
RUN apt-get install -y nvidia-container-toolkit

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/program/Real-ESRGAN/
ENV SAGEMAKER_PROGRAM serve

WORKDIR /opt/program/Real-ESRGAN
