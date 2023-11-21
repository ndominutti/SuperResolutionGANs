# Deploy

<br>

## Overview

This phase involves retrieving the trained model weights stored in an S3 bucket. Once retrieved, the deployment process initiates by creating a model using these weights. Finally, a real-time inference endpoint is launched, powered by FastAPI, running on a CUDA-based instance to facilitate efficient and high-speed model predictions.

<br>

## Structure

* **Dockerfile**: the dockerfile for the inference container
* **serve**: sh commands to launch de uvicorn server and open the port 8080
* **requirements.txt**: required libraries for the inference
* **predictor.py**: main file with de FastAPI code
* **main.py**: FastAPI launcher file
* **scripts/**: DIR with the __ init __.py file that will override the one used in the [Real-ESRGAN repo](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/__init__.py) and a utils file to read the config YAML
* **config/**: contains the configuration file that rules the whole process

<br>

## Hints

* You can control almost any parameter from the config YAML file
* Do not try to run inference in a non GPU based instance as you would not be able to use half precision (fp16), instead use an instance with a GPU enabled (ml.g4dn.xlarge will do fine)
* To perform invocations you can use the boto3 API

```
client = boto3.client('sagemaker')
...
```