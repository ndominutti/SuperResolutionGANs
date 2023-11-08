#!/bin/bash

# Real ESRNET
python realesrgan/train.py -opt options/custom_realesrnet_training.yml --debug


# Real ESRGAN
# python3 realesrgan/train.py -opt options/custom_realesrgan_training.yml --auto_resume