FROM nvidia/cuda:12.1.1-base-ubuntu20.04 
# 12.1.1-runtime-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get -y update && apt-get install -y --no-install-recommends\  
 python3-pip\
 git\
 libgl1-mesa-glx\
 libglib2.0-0 \
 curl 

WORKDIR /opt/program
#Run instalations
COPY requirements.txt .
RUN pip3 install -r requirements.txt

#Clone main repo
RUN git clone https://github.com/xinntao/Real-ESRGAN.git
#Create a minimum structure
RUN mkdir /opt/program/config
# Copy key files to the container
COPY scripts/__init__.py Real-ESRGAN/realesrgan
COPY scripts/utils.py Real-ESRGAN/.
COPY config/config.yml /opt/program/config
COPY main.py Real-ESRGAN/.
COPY predictor.py Real-ESRGAN/.
COPY serve Real-ESRGAN/.


#Install nvidia toolkit
RUN apt-get install -y gpg sed
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    apt-get update
RUN apt-get install -y nvidia-container-toolkit


ENV PATH="/opt/program/Real-ESRGAN:${PATH}"
WORKDIR /opt/program/Real-ESRGAN
