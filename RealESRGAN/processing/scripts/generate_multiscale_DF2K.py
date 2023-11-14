import argparse
import glob
import os
from PIL import Image
import gc
from joblib import Parallel, delayed

def process_image(path, scale, output_folder):
    print(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path)
    width, height = img.size
    new_width, new_height = int(width * scale), int(height * scale)
    rlt = img.resize((new_width, new_height), resample=Image.LANCZOS)
    output_path = os.path.join(output_folder, f'{basename}T{scale:.2f}.png')
    rlt.save(output_path)
    del rlt
    del img
    gc.collect()

def main(args):
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400
    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    output_folder = args.output

    Parallel(n_jobs=-1, backend="threading")(
        delayed(process_image)(path, scale, output_folder)
        for path in path_list
        for scale in scale_list
    )

    # Save the smallest image for each original image
    Parallel(n_jobs=-1, backend="threading")(
        delayed(process_image)(path, scale_list[-1], output_folder)
        for path in path_list
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input folder')
    parser.add_argument('--output', type=str, help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
