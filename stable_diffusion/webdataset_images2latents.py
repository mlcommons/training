import os
import tarfile
import argparse
import tempfile

import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
import torchvision
from torchvision import transforms

from ldm.data.utils import rearrange_transform
from ldm.util import instantiate_from_config


image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".ico"]

def is_image(filename):
    return any(filename.endswith(ext) for ext in image_extensions)


def process_image(input_image_path, output_tensor_name, image_transforms, model):
    original_image = Image.open(input_image_path).convert('RGB')
    transformed_img = image_transforms(original_image).float().unsqueeze(0).to(model.device)
    encoded_image = model.encode_first_stage(transformed_img).sample().squeeze(0)
    np.save(output_tensor_name, encoded_image.to("cpu").numpy())


def process_tar(input_tar, output_tar, image_transforms, model):
    with tempfile.TemporaryDirectory() as tempdir:
        # Extract the input tar into a temporary folder
        with tarfile.open(input_tar, 'r') as tar:
            tar.extractall(path=tempdir)

        # Walk through all the files in the directory
        for subdir, dirs, files in os.walk(tempdir):
            for file in tqdm(files, desc=f'Processing files', unit='file'):
                file_path = os.path.join(subdir, file)
                if is_image(file_path):
                    file_path_without_ext = os.path.splitext(file_path)[0]
                    process_image(file_path, file_path_without_ext + '.npy', image_transforms, model)
                    os.remove(file_path)  # remove the original image file

        # Recreate the tarfile with the modified images.
        # for atomicity, we first create a temp tar then rename it
        temp_tar_file = output_tar + '.tmp'
        with tarfile.open(temp_tar_file, 'w') as tar:
            tar.add(tempdir, arcname='')
        os.rename(temp_tar_file, output_tar)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    print(f"instantiate_from_config")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and resize images in tar files.')
    parser.add_argument('--input-tar', required=True, help='Input tar file')
    parser.add_argument('--output-tar', required=True, help='Output tar file')
    parser.add_argument('--config', required=True, help='The model config')
    parser.add_argument('--ckpt', required=True, help='The model config')
    parser.add_argument('--resolution', default=512, help='Output image resolution')

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Image transform
    image_transforms = transforms.Compose([
        torchvision.transforms.ToTensor(), # (H x W x C) -> (C x H x W).
        torchvision.transforms.Resize(size=args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(size=args.resolution),
    ])

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt).to(device)

    process_tar(args.input_tar, args.output_tar, image_transforms, model)
