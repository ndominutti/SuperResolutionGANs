import boto3
from PIL import Image
import io
import argparse

def send_request(args):
    """
    Send a request to the InService sagemaker Endpoint
    """
    endpoint_name = args.endpoint_name
    image_path = args.image_path
    
    client = boto3.client('sagemaker-runtime', region_name='us-east-2')
    with open(image_path, 'rb') as f:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='image/png', 
            Body=f.read()
        )
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        # Assuming it's a PNG image
        image = Image.open(io.BytesIO(response['Body'].read()))
        image.save(args.image_save_path)
        print('Image saved correctly!')
    else:
        print(f"Request failed with status code: {response['ResponseMetadata']['HTTPStatusCode']}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str, default='RealESRGAN-Endpoint')
    parser.add_argument("--image-path", type=str, default='./sample_img/sample_500.png')
    parser.add_argument("--image-save-path", type=str, default='./recieved_img.png')
    send_request(parser.parse_args())    
