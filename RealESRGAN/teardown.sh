#!/usr/bin/env bash


echo "Deleting S3 bucket..."
aws s3 rm s3://realesrgan/train/ --recursive
aws s3 rm s3://realesrgan/validation/ --recursive
aws s3 rm s3://realesrgan/lambda_function/ --recursive

echo "Deleting cloudformation stacks..."
aws cloudformation delete-stack --stack-name RealESRGANBucketStack
aws cloudformation delete-stack --stack-name RealESRGANStack

echo "Deleting ECR images..."
aws ecr delete-repository --repository-name resrgan_processing_job_image
aws ecr delete-repository --repository-name resrgan_training_job_image
aws ecr delete-repository --repository-name  resrgan_inference_image

echo "Deleting Sagemaker model..."
aws sagemaker delete-model --model-name RealESRGAN-Model-19H-17M-52S 
echo "Deleting Sagemaker packages & packages group..."
packages=$(aws sagemaker list-model-packages --model-package-group-name RealESRGAN --output text | awk '{print $4}')
for package in $packages; do
    aws sagemaker delete-model-package --model-package-name $package
done
aws sagemaker delete-model-package-group --model-package-group-name RealESRGAN
echo "Deleting Sagemaker endpoint configuration..."
aws sagemaker delete-endpoint-config --endpoint-config-name RealESRGAN-Endpoint-Config-20H-31M-27S
echo "Deleting Sagemaker endpoint..."
aws sagemaker delete-endpoint --endpoint-name RealESRGAN-Endpoint