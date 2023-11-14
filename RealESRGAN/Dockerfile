FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get -y update && apt-get install -y --no-install-recommends\  
 python3-pip\
 wget\
 git\
 libgl1-mesa-glx\
 libglib2.0-0 

WORKDIR /opt/program
RUN git clone https://github.com/xinntao/Real-ESRGAN.git
COPY requirements.txt Real-ESRGAN/
RUN pip3 install -r Real-ESRGAN/requirements.txt


#For training the Real-ESRGAN
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

# COPY scripts/lowres.py Real-ESRGAN/scripts
# COPY scripts/generate_multiscale_DF2K.py Real-ESRGAN/scripts
COPY config/custom_finetune_4x.yml Real-ESRGAN/options/

ENV PATH="/opt/program/Real-ESRGAN:${PATH}"
ENV TENSORBOARD_LOGS_PATH='/opt/ml/output/tensorboard/'

COPY scripts/train.py Real-ESRGAN/realesrgan
COPY train Real-ESRGAN
# COPY deploy Real-ESRGAN
WORKDIR /opt/program/Real-ESRGAN
