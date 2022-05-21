#!/usr/bin/env python

import os
import json
import argparse
import fiftyone as fo
import fiftyone.zoo as foz

parser = argparse.ArgumentParser(description='Download OpenImages using FiftyOne', add_help=True)
parser.add_argument('--dataset-dir', default='/open-images-v6', help='dataset download location')
parser.add_argument('--splits', default=['train', 'validation'], choices=['train', 'validation', 'test'],
                    nargs='+', type=str,
                    help='Splits to download, possible values are train, validation and test')
parser.add_argument('--classes', default=None, nargs='+', type=str,
                    help='Classes to download. default to all classes')
parser.add_argument('--output-labels', default='labels.json', type=str,
                    help='Classes to download. default to all classes')
args = parser.parse_args()


print("Downloading open-images dataset ...")
dataset = foz.load_zoo_dataset(
    name="open-images-v6",
    classes=args.classes,
    splits=args.splits,
    label_types="detections",
    dataset_name="open-images",
    dataset_dir=args.dataset_dir
)

print("Converting dataset to coco format ...")
for split in args.splits:
    output_fname = os.path.join(args.dataset_dir, split, "labels", args.output_labels)
    split_view = dataset.match_tags(split)
    split_view.export(
        labels_path=output_fname,
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections",
        classes=args.classes)

    # Add iscrowd label to openimages annotations
    with open(output_fname) as fp:
        labels = json.load(fp)
    for annotation in labels['annotations']:
        annotation['iscrowd'] = int(annotation['IsGroupOf'])
    with open(output_fname, "w") as fp:
        json.dump(labels, fp)
