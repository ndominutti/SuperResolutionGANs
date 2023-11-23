#!/usr/bin/env bash

#Params:
#    stack_behavior(string): controls wheter to create the cloudformation stack or #if only an update is needed.
#    Have in mind that when an stack is already created you MUST run an update, for #creation will fail. Accepted
#    values: update or create

stack_behavior=$1 

echo "Installing requirements"
#pip3 install -r requirements.txt

echo "Running S3 bucket creation..."
#if [ "$stack_behavior" = "create" ]; then
    #aws cloudformation create-stack --stack-name RealESRGANBucketStack \
    #    --template-body file://infrastructure/BucketCreationTemplate.yml
#elif [ "$stack_behavior" = "update" ]; then
    #aws cloudformation update-stack --stack-name RealESRGANBucketStack \
    #    --template-body file://infrastructure/BucketCreationTemplate.yml
# else
#     echo "Invalid stack behavior. Usage: $0 <create/update>"
# fi

echo "Zipping lambda function and uploading into S3..."
cd infrastructure/lambda_function
zip ../lambda_code.zip ./*
aws s3 cp ../lambda_code.zip s3://real-esrgan/lambda_function/
cd ../..

echo "Running cloudformation stack building..."
if [ "$stack_behavior" = "create" ]; then
    aws cloudformation create-stack --stack-name RealESRGANStack \
        --template-body file://infrastructure/RealESRGANTemplate.yml \
        --capabilities CAPABILITY_IAM
elif [ "$stack_behavior" = "update" ]; then
    aws cloudformation update-stack --stack-name RealESRGANStack \
        --template-body file://infrastructure/RealESRGANTemplate.yml \
        --capabilities CAPABILITY_IAM
else
    echo "Invalid stack behavior. Usage: $0 <create/update>"
fi

#######
# WATCHOUT IMAGES NAME HAS CHANGED, CHANGE THE PIPELINE

#echo "Building and pushing Preprocess image into ECR..."
#./build_and_push.sh resrgan_processing_job_image preprocess/.

#echo "Building and pushing Training image into ECR..."
#./build_and_push.sh resrgan_training_job_image train/.

#echo "Building and pushing Deploy image into ECR..."
#./build_and_push.sh resrgan_inference_image deploy/.