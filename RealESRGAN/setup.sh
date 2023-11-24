#!/usr/bin/env bash

#Params:
#    stack_behavior(string): controls wheter to create the cloudformation stack or #if only an update is needed.
#    Have in mind that when an stack is already created you MUST run an update, for #creation will fail. Accepted
#    values: update or create

stack_behavior=$1 

echo "Running S3 bucket creation..."
if [ "$stack_behavior" = "create" ]; then
    aws cloudformation create-stack --stack-name RealESRGANBucketStack \
       --template-body file://infrastructure/BucketCreationTemplate.yml
elif [ "$stack_behavior" = "update" ]; then
    aws cloudformation update-stack --stack-name RealESRGANBucketStack \
       --template-body file://infrastructure/BucketCreationTemplate.yml
else
    echo "Invalid stack behavior. Usage: $0 <create/update>"
fi


echo "Installing requirements"
pip3 install -r requirements.txt


echo "Zipping lambda function and uploading into S3..."
cd infrastructure/lambda_function
zip ../lambda_code.zip ./*
aws s3 cp ../lambda_code.zip s3://real-esrgan/lambda_function/
cd ../..


echo "Running cloudformation stack building..."
if [ "$stack_behavior" = "create" ]; then
    aws cloudformation create-stack --stack-name RealESRGANStack \
        --template-body file://infrastructure/RealESRGANTemplate.yml \
        --capabilities CAPABILITY_NAMED_IAM
elif [ "$stack_behavior" = "update" ]; then
    aws cloudformation update-stack --stack-name RealESRGANStack \
        --template-body file://infrastructure/RealESRGANTemplate.yml \
        --capabilities CAPABILITY_NAMED_IAM
else
    echo "Invalid stack behavior. Usage: $0 <create/update>"
fi


echo "Running S3 bucket schema generation..."
region=$(aws configure get region)
while true; do
    status=$(aws cloudformation describe-stacks --stack-name "RealESRGANBucketStack" --region "$region" --query "Stacks[0].StackStatus" --output text)
    
    if [[ $status == "CREATE_COMPLETE" ]]; then
        break
    elif [[ $status == "ROLLBACK_COMPLETE" || $status == "CREATE_FAILED" || $status == "ROLLBACK_FAILED" ]]; then
        exit 1
    else
        sleep 5
    fi
done
aws s3api put-object --bucket real-esrgan --key train/
aws s3api put-object --bucket real-esrgan --key train/hq/
aws s3api put-object --bucket real-esrgan --key train/lq/
aws s3api put-object --bucket real-esrgan --key validation/
aws s3api put-object --bucket real-esrgan --key validation/hq/
aws s3api put-object --bucket real-esrgan --key validation/lq/


echo "Building and pushing Preprocess image into ECR..."
./build_and_push.sh resrgan_processing_job_image preprocess/.
echo "Removing local image to save space..."
docker images --format '{{.Repository}}:{{.Tag}}' | grep 'resrgan_processing_job_image' | awk '{print $1}' | xargs -I {} docker rmi {}

echo "Building and pushing Training image into ECR..."
./build_and_push.sh resrgan_training_job_image train/.
echo "Removing local image to save space..."
docker images --format '{{.Repository}}:{{.Tag}}' | grep 'resrgan_training_job_image' | awk '{print $1}' | xargs -I {} docker rmi {}

echo "Building and pushing Deploy image into ECR..."
./build_and_push.sh resrgan_inference_image deploy/.
echo "Removing local image to save space..."
docker images --format '{{.Repository}}:{{.Tag}}' | grep 'resrgan_inference_image' | awk '{print $1}' | xargs -I {} docker rmi {}
