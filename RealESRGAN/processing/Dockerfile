FROM python:3.9

RUN apt-get -y update && apt-get install -y --no-install-recommends python3-pip libgl1-mesa-glx libglib2.0-0


WORKDIR /opt/ml/processing
COPY scripts/lowres.py .
COPY scripts/generate_multiscale_DF2K.py .
COPY scripts/generate_meta_info.py .
COPY requirements.txt .
COPY process .

RUN pip3 install -r requirements.txt
RUN chmod +x process

ENTRYPOINT ["./process"]






 