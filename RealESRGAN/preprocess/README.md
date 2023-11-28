# Processing job

<br>

## Overview

This step involves reading the high-quality (HQ) training and validation images stored in S3 and generating a lower-quality (LQ) version. This LQ version is utilized during the training phase. Separating this step from the training job serves the dual purpose of maintaining modularity and optimizing costs. Unlike the training job, the processing job doesn't require GPU resources, contributing to cost efficiency.

<br>

## Structure

* **Dockerfile**: the dockerfile for the processing container
* **process**: sh commands to perform the processing job
* **requirements.txt**: required libraries for the job
* **scripts/**: DIR with some preprocessing files from the [here](https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/generate_multiscale_DF2K.py) and [here](https://github.com/xinntao/Real-ESRGAN/blob/master/scripts/generate_meta_info.py) with some minor changes.


<br>

## Hints

* For this step an instance with 8GB memory size will do, no GPU needed (ml.m5.large will do fine)