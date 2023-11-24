# Real ESRGAN | Finetune and deploy in AWS Sagemaker

<br>

## Overview
A Real ESRGAN is a model used for Super Resolution tasks, it was proposed in [Xintao Wang et al. 2021. PReal-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833) and implemented in [this repository](https://github.com/xinntao/Real-ESRGAN).


Here I present a particular implementation of a fine-tuning and deploying process using AWS Sagemaker, this includes a:
* Processing Job: in charge of creating the training set, using a High Quality set of images stored in S3
* Training Job: training the model
* Inference Endpoint creation: running an inference server in a sagemaker endpoint

<br>

For this case, I've used the excellent work done in the official [model implementation](https://github.com/xinntao/Real-ESRGAN) and in the [Basiscr module](https://github.com/XPixelGroup/BasicSR) as the foundation. Several modifications have been applied to these modules:

* In the BasicSR module, I forked the repository and made minor adjustments, primarily related to the PATH where the logs are saved. This alteration enables TensorBoard to locate these logs during the execution of the SageMaker Training job.
* Regarding the Real-ESRGAN module, most of the changes are stored in this repository within the /script directories for each process. These modified scripts are copied inside the Dockerfile, replacing the original ones

<br>

## Sample result

This use case utilizes an scaling ratio of 1:4, where the image size is incremented in 4 times:
<div style="text-align: center;">
    <img src="static/results.png" width="560" height="430" />
</div>

One might argue that this could be achieved with a single interpolation. However, upon zooming into the image, it becomes evident that incrementing the image size doesn't reduce its quality; instead, it enhances it

![Comparison of input and output image](static/results.gif)

<br>

## Repository structure

* **setup.sh**: startup file, in charge of setting the needed infrastructure to run the full pipeline
* **build_and_push.sh**: this is an useful file to build and push images to ECR
* **infrastructure/**: in this DIR you will find the cloudformation + lambda code
* **preprocess/**: in this DIR you will find the processing job to preprocess the images before training
* **train/**: in this DIR you will find the training job
* **deploy/**: in this DIR you will find the necessary code to create a model with the trained model and deploying it into a sagemaker endpoint
* **pipeline/**: in this DIR you will find the implementation of a Sagemaker pipeline that includes the end to end job, including a model validation step after training to notify and allow a human to approve the training metrics before deploying

<br>

## Usage
Install docker
Remember about quotas for jobs
Have in mind the needed space
For running the setup.sh file at least 8GB RAM are needed, otherwise the process will crash
```
#Setup the infrastructure
chmod +x setup.sh
./setup.sh create

# Copy your training images into S3://real-esrgan/train/HQ and your validation images into S3://real-esrgan/validation/HQ

# Initiate pipeline excecution
python3 pipeline/pipeline.py

# Approve the model


```

<br>

## Cleanup

```
# Delete bucket subfolders
aws s3 rm s3://real-esrgan/train/ --recursive
aws s3 rm s3://real-esrgan/validation/ --recursive
aws s3 rm s3://real-esrgan/lambda_function/ --recursive

# Delete cloudformation stack
aws cloudformation delete-stack --stack-name RealESRGANBucketStack
aws cloudformation delete-stack --stack-name RealESRGANStack

# Delete ECR images
aws ecr batch-delete-image --repository-name resrgan_processing_job_image
aws ecr batch-delete-image --repository-name resrgan_training_job_image
aws ecr batch-delete-image --repository-name  resrgan_inference_image

# Delete model, endpoint config and endpoint
aws sagemaker delete-model --model-name RealESRGAN-Model
aws sagemaker delete-endpoint-config --endpoint-config-name <CHECK THE ENDPOINT CONFIG NAME>
aws sagemaker delete-endpoint --endpoint-name RealESRGAN-Endpoint
```




