FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get -y update && apt-get install -y --no-install-recommends\  
 python3-pip\
 wget\
 git\
 libgl1-mesa-glx\
 libglib2.0-0 

WORKDIR /opt/program
RUN git clone https://github.com/xinntao/Real-ESRGAN.git
# COPY custom_realesrnet_training.yml Real-ESRGAN/options/
COPY requirements.txt Real-ESRGAN/
COPY lowres.py Real-ESRGAN/scripts
RUN pip3 install -r Real-ESRGAN/requirements.txt
#COPY train/ /opt/ml/input/data/training


#For training the Real-ESRGAN
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

COPY custom_finetune_4x.yml Real-ESRGAN/options/

ENV PATH="/opt/program/Real-ESRGAN:${PATH}"
ENV TENSORBOARD_LOGS_PATH='/opt/ml/output/tensorboard/'

COPY train.py Real-ESRGAN/realesrgan
COPY model Real-ESRGAN
WORKDIR /opt/program/Real-ESRGAN
