from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
from config import *
import sagemaker
import time

PIPELINE_SESSION = PipelineSession()

def main():
    # PROCESSING JOB
    print('*'*10 + ' RUNNING PROCESSING JOB DEFINITION ' + '*'*10)
    processor = Processor(
        role=ROLE,
        image_uri=PROCESSING_JOB_IMAGE,
        instance_count=INSTANCE_COUNT,
        instance_type=PROCESSING_JOB_INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE,
        base_job_name=PROCESSING_JOB_NAME,
        sagemaker_session=PIPELINE_SESSION
    )
    
    processing_args = processor.run(
        inputs=[ProcessingInput(source="s3://realesrgan/train/hq/", destination="/opt/ml/processing/data/training/hq"),
               ProcessingInput(source="s3://realesrgan/validation/hq/", destination="/opt/ml/processing/data/validation/hq")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/data/training/lq",destination="s3://realesrgan/train/lq/"),
                 ProcessingOutput(source="/opt/ml/processing/data/training/meta_info",destination="s3://realesrgan/train/"),
                ProcessingOutput(source="/opt/ml/processing/data/validation/lq",destination="s3://realesrgan/validation/lq/")]
    )
    
    processing_step = ProcessingStep(
        name='RealESRGAN-preprocess',
        step_args=processing_args
    )
    
    #TRAINING JOB
    print('*'*10 + ' RUNNING TRAINING JOB DEFINITION ' + '*'*10)
    TBC = TensorBoardOutputConfig(
        s3_output_path=TRAINING_JOB_OUTPUT_PATH
    )
    
    estimator = sagemaker.estimator.Estimator(
        image_uri=TRAINING_JOB_IMAGE,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=TRAINING_JOB_INSTANCE_TYPE,
        output_path=TRAINING_JOB_OUTPUT_PATH,
        sagemaker_session=PIPELINE_SESSION,
        container_port=TRAINING_JOB_CONTAINER_PORT,
        tensorboard_output_config=TBC,
        base_job_name=TRAINING_JOB_NAME
        
    )
    
    training_args = estimator.fit(
        inputs={
            'training': TrainingInput(
                s3_data=TRAINING_JOB_TRAIN_INPUT_PATH
            ),
            'validation': TrainingInput(
                s3_data=TRAINING_JOB_VALIDATION_INPUT_PATH
            )
        }
    )
    
    training_step = TrainingStep(
        name='RealESRGAN-train',
        step_args=training_args,
        depends_on=[processing_step]
    )
    
    #MODEL REGISTRATION
    print('*'*10 + ' RUNNING MODEL REGISTRATION DEFINITION ' + '*'*10)
    model = sagemaker.Model(
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=ROLE,
            image_uri=INFERENCE_JOB_IMAGE,
            sagemaker_session=PIPELINE_SESSION 
        )
    
    register_model_step_args = model.register(
       content_types=["image/png","image/jpeg"],
        response_types=["image/png","image/jpeg"],
       inference_instances=[INFERENCE_JOB_INSTANCE_TYPE],
        approval_status=   MODEL_APPROVAL_STATUS,
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME
    )
    
    model_registration_step = ModelStep(
       name="RealESRGAN-RegisterModel",
       step_args=register_model_step_args,
       depends_on=[training_step]
    )
    
    #PIPELINE DEFINITION
    print('*'*10 + ' RUNNING PIPELINE DEFINITION ' + '*'*10)
    pipeline = Pipeline(
        name='RealESRGAN-Pipeline',
        steps=[processing_step, training_step, model_registration_step],
        parameters=[
            PROCESSING_JOB_IMAGE,
            INSTANCE_COUNT,
            PROCESSING_JOB_INSTANCE_TYPE,
            VOLUME_SIZE,
            TRAINING_JOB_IMAGE,
            TRAINING_JOB_OUTPUT_PATH,
            TRAINING_JOB_INSTANCE_TYPE,
            TRAINING_JOB_CONTAINER_PORT,
            TBC,
            TRAINING_JOB_TRAIN_INPUT_PATH,
            TRAINING_JOB_VALIDATION_INPUT_PATH,
            INFERENCE_JOB_IMAGE,
            INFERENCE_JOB_INSTANCE_TYPE,
            MODEL_APPROVAL_STATUS,
            MODEL_PACKAGE_GROUP_NAME
    ]
    )

    print('*'*10 + ' STARTING PIPELINE EXECUTION ' + '*'*10)
    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    execution.wait()

if __name__=='__main__':
    main()