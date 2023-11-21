import numpy as np
import cv2
import argparse
import glob
import os
from PIL import Image
import math

def main(args:dict) -> None:
    """
    Implement a resolution downgrade converting the input images into 1/4 of their
    original size.

    Args:
        args(dict): arguments from argparse.ArgumentParser()

    Returns:
        None
    """
    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        w,h,_ = img.shape
        jpeg_quality = 10
        _, jpeg_data = cv2.imencode('.jpeg', img)#, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        jpeg_img = cv2.imdecode(jpeg_data, cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(jpeg_img, (math.floor(h/4), math.floor(w/4)), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(args.output, f'{basename}.png'), resized_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input folder')
    parser.add_argument('--output', type=str, help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)