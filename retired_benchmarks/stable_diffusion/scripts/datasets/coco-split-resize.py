#!/usr/bin/env python3

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input-images-dir", type=str, required=True)
parser.add_argument("--input-captions-file", type=str, required=True)
parser.add_argument("--output-images-dir", type=str, required=True)
parser.add_argument("--output-tsv-file", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=30000)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--allow-duplicate-images", type=bool, default=False)

args = parser.parse_args()


def resize_image(input_image, output_image, width, height, resample=Image.Resampling.BICUBIC):
    print(f"{input_image} -> {output_image}")
    image = Image.open(input_image)
    image = image.resize((width, height), resample=resample)
    image.save(output_image)


# Load coco annotations
with open(args.input_captions_file, "r") as f:
    captions = json.load(f)
    annotations = captions["annotations"]

# Convert to dataframe
df = pd.DataFrame(annotations)
df['caption'] = df['caption'].apply(lambda x: x.replace('\n', '').strip())

# Shuffle the dataframe
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Keep a single captions per image
if not args.allow_duplicate_images:
    df = df.drop_duplicates(subset=["image_id"], keep="first")

# Take a subset
df = df[:args.num_samples]

# Sort by id
df = df.sort_values(by=["id"])

# Save the subset to a tsv file
df.to_csv(args.output_tsv_file, sep="\t", index=False)

# Create output image directory if it doesn't exist
os.makedirs(args.output_images_dir, exist_ok=True)

# resize images with a worker pool
with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    for i, row in df.iterrows():
        image_fname = f"COCO_val2014_{row['image_id']:012}.jpg"
        input_img = os.path.join(args.input_images_dir, image_fname)
        output_img = os.path.join(args.output_images_dir, image_fname)

        executor.submit(resize_image, input_img, output_img, args.width, args.height, Image.Resampling.BICUBIC)
