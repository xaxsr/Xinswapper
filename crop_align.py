import os
import cv2
import torch
import onnxruntime
import numpy as np

from tqdm import tqdm
from pathlib import Path

from util.cropping import detect_retinaface, norm_crop

def crop_align(retinaface_model, input_dir, output_dir, extension, quality, crop_size):
    image_files = [os.path.join(input_dir, os.path.basename(f)) for f in Path(input_dir).glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    syncvec = torch.empty((1, 1), dtype=torch.float32, device='cuda:0')

    print(f">process {len(image_files)} images...")
    faces_detected = 0

    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        img = cv2.imread(image_file)
        base_name = os.path.splitext(os.path.basename(image_file))[0]

        if img is not None:
            img = torch.from_numpy(img.astype('uint8')).to('cuda')

            pad_scale = 0.2
            padded_width = int(img.size()[1] * (1. + pad_scale))
            padded_height = int(img.size()[0] * (1. + pad_scale))

            padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device='cuda:0')

            width_start = int(img.size()[1] * pad_scale / 2)
            width_end = width_start + int(img.size()[1])
            height_start = int(img.size()[0] * pad_scale / 2)
            height_end = height_start + int(img.size()[0])

            padding[height_start:height_end, width_start:width_end, :] = img
            img = padding

            img = img.permute(2, 0, 1)

            try:
                kpss = detect_retinaface(retinaface_model=retinaface_model, img=img, max_num=3, score=0.4, syncvec=syncvec)
            except IndexError:
                pass
            else:
                for j, kps in enumerate(kpss):
                    img_np = norm_crop(img.cpu().numpy().transpose(1, 2, 0), kps, crop_size)
                    output_path = os.path.join(output_dir, f'{base_name}_{j:05d}_.{extension}')
                    if extension == 'jpg':
                        cv2.imwrite(output_path, img_np.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                    else:
                        cv2.imwrite(output_path, img_np.astype(np.uint8))
                    faces_detected += 1

    print(f">done! found {faces_detected} faces")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./raw_images/', help="path to raw images")
    parser.add_argument('--output_dir', type=str, default='./raw_images/aligned/', help="path to save aligned faces")
    parser.add_argument('--output_ext',
                        type=lambda value: value.lower() if value.lower() in ['jpg','png'] else argparse.ArgumentTypeError("output_ext must be 'jpg' or 'png'"),
                        default='jpg',
                        help="output image extension ('jpg' or 'png')")
    parser.add_argument('--jpg_quality', type=int, default=100, help="compression quality (for 'jpg' only)")
    parser.add_argument('--crop_size', type=int, default=1024, help="crop size")
    parser.add_argument('--model_path', type=str, default='./weights/det_10g.onnx', help="path to retinaface model (det_10g.onnx)")

    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if os.path.isfile(args.model_path):
            retinaface_model = onnxruntime.InferenceSession(args.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            crop_align(retinaface_model, args.input_dir, args.output_dir, args.output_ext.lower(), args.jpg_quality, args.crop_size)
        else:
            print(f">cannot find retinaface model at {args.model_path}")
    else:
        print(">input directory not valid")