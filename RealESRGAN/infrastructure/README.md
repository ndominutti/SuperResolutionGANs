# Infrastructure

<br>

## Overview

Here you will find the CloudFormation templates to set the needed infrastructure. Have in mind that it's advised to run the infrastructure setup from the _../setup.sh_ file, as it contains the templates executions + needed bash commands.

<br>

## Structure
* **BucketCreationTemplate.yml**: CloudFormation template to create the main RealESRGAN bucket
* **RealESRGANTemplate.yml**: CloudFormation template to create the rest of the needed infrastructure
* **lambda_function/**: this DIR contains the needed code for the lambda function