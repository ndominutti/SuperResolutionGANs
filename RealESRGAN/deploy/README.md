# Deploy

<br>

## Overview

This DIR contains the code of the inference endpoint image and the FastAPI app code. 

<br>

## Structure

* **Dockerfile**: the dockerfile for the inference container
* **serve**: sh commands to launch de uvicorn server and open the port 8080
* **requirements.txt**: required libraries for the inference
* **predictor.py**: main file with de FastAPI code
* **main.py**: FastAPI launcher file
* **scripts/**: DIR with the __ init __.py file that will override the one used in the [Real-ESRGAN repo](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/__init__.py) and a utils file to read the config YAML
* **config/**: contains the configuration file that rules the whole process
* **endpoit_request.py**: this is a script with the sample code to perform a request to the running endpoint and save it locally

<br>

## Hints

* You can control almost any parameter from the config YAML file
* Do not try to run inference in a non GPU based instance as you would not be able to use half precision (fp16), instead use an instance with a GPU enabled (ml.g4dn.xlarge will do fine)