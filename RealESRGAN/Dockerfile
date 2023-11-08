FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get -y update && apt-get install -y --no-install-recommends python3-pip\
 wget\
 git

RUN git clone https://github.com/xinntao/Real-ESRGAN.git
COPY custom_realesrnet_training.yml Real-ESRGAN/options/
COPY train.sh Real-ESRGAN
# RUN cp custom_realesrgan_training.yml Real-ESRGAN/options/

RUN cd Real-ESRGAN
#For training the Real-ESRNet
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth \
                                                                                            -P experiments/pretrained_models



# RUN chmod +x train.sh
# RUN ./train.sh
