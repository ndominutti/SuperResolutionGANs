import boto3

my_session = boto3.session.Session()
REGION = my_session.region_name
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
MODEL_NAME = 'RealESRGAN-Model-'
INFERENCE_JOB_INSTANCE_TYPE = "ml.g4dn.xlarge"
INSTANCE_COUNT=1
ENDPOINT_NAME = 'RealESRGAN-Endpoint'
ENDPOINT_CONFIG_NAME = 'RealESRGAN-Endpoint-Config-'
MODEL_PACKAGE_NAME = 'RealESRGAN'
EXECUTION_ROLE = f"arn:aws:iam::{ACCOUNT}:role/RealESRGANRole"
