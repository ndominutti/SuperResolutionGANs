import json
import boto3
import config
import time
from datetime import datetime

def lambda_handler(event, context):
    """ 
    Create a SM model + endpoint config + endpoint.
    If the model endpoint already exists, it will be updated with a new endpoint config
    """
    date = datetime.now().strftime("%HH-%MM-%SS")
    MODEL_NAME           = config.MODEL_NAME + date
    ENDPOINT_CONFIG_NAME = config.ENDPOINT_CONFIG_NAME + date
    
    sm_client = boto3.client("sagemaker")
    create_model_respose = sm_client.create_model(ModelName=MODEL_NAME, 
                              PrimaryContainer={
                                  'ModelPackageName':f'arn:aws:sagemaker:{config.REGION}:{config.ACCOUNT}:model-package/{config.MODEL_PACKAGE_NAME}/{event["detail"]["ModelPackageVersion"]}'
                              },
                              ExecutionRoleArn=config.EXECUTION_ROLE)
    
    
    
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "InstanceType": config.INFERENCE_JOB_INSTANCE_TYPE,
                "InitialInstanceCount": config.INSTANCE_COUNT,
                "ModelName": MODEL_NAME,
                "VariantName": "AllTraffic",
            }
        ]
    )
    try:
        create_endpoint_response = sm_client.create_endpoint(EndpointName=config.ENDPOINT_NAME, 
                                                             EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print('*'*30 + ' Model endpoint CREATED successfuly! ' + '*'*30)
    except:
        update_endpoint_response = sm_client.update_endpoint(EndpointName=config.ENDPOINT_NAME, 
                                                             EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print('*'*30 + ' Model endpoint UPDATED successfuly! ' + '*'*30)
    
    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
        "other_key": "example_value",
    }
            