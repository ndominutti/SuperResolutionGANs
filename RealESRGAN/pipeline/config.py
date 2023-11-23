import sagemaker
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

SESSION = sagemaker.Session()
ACCOUNT = SESSION.boto_session.client("sts").get_caller_identity()["Account"]
ROLE = f"arn:aws:iam::{ACCOUNT}:role/RealESRGANRole"
REGION = SESSION.boto_session.region_name
INSTANCE_COUNT=ParameterInteger(name="InstanceCount", default_value=1)
#############################################
PROCESSING_JOB_IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/resrgan_processing_job_image:latest"
PROCESSING_JOB_INSTANCE_TYPE = ParameterString(name="ProcessingJobInstanceType", default_value="ml.m5.large")
PROCESSING_JOB_NAME = 'RealESRGAN-preprocessing'
VOLUME_SIZE = ParameterInteger(name="ProcessingVolumeSize", default_value=20)
#############################################
TRAINING_JOB_IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/resrgan_training_job_image:latest"
TRAINING_JOB_INSTANCE_TYPE = ParameterString(name="TrainingJobInstanceType", default_value="ml.g4dn.xlarge")
TRAINING_JOB_NAME = 'RealESRGAN-training'
TRAINING_JOB_OUTPUT_PATH = ParameterString(name="TrainingJobOutputPath", default_value="s3://{}/output".format(SESSION.default_bucket()))
TRAINING_JOB_TRAIN_INPUT_PATH = ParameterString(name="TrainingJobTrainInputPath", default_value="s3://real-esrgan/train")
TRAINING_JOB_TRAIN_VALIDATION_PATH = ParameterString(name="TrainingJobValidationInputPath", default_value="s3://real-esrgan/validation")
TRAINING_JOB_CONTAINER_PORT = ParameterInteger(name="TrainingJobContainerPort", default_value=8080)
#############################################
INFERENCE_JOB_IMAGE = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/resrgan_inference_image:latest"
INFERENCE_JOB_INSTANCE_TYPE = "ml.g4dn.xlarge"
MODEL_NAME = ParameterString(name="ModelName", default_value='RealESRGAN')
MODEL_APPROVAL_STATUS    = "PendingManualApproval"
MODEL_PACKAGE_GROUP_NAME = "RealESRGAN"