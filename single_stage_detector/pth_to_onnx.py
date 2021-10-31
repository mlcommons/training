#!/usr/bin/env python3
import argparse

import torch
import torch.onnx
import torchvision
from torch.autograd import Variable

from model.retinanet import (retinanet_resnet50_fpn,
                             retinanet_resnext50_32x4d_fpn)


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Convert PyTorch detection file to onnx format', add_help=add_help)

    parser.add_argument('--input', required=True, help='input pth file')
    parser.add_argument('--output', default=None, help='output onnx file')

    parser.add_argument('--backbone', default='resnext50_32x4d_fpn', choices=['resnet50_fpn', 'resnext50_32x4d_fpn'],
                        help='The model backbone')
    parser.add_argument('--num-classes', default=91, type=int,
                        help='Number of detection classes')

    parser.add_argument('--image-size', default=None, nargs=2, type=int,
                        help='Image size for training. If not set then will be dynamic')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='input batch size. if not set then will be dynamic')
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()

    args.output = args.output or ('retinanet_'+args.backbone+'.onnx')
    return args

def main(args):
    batch_size = args.batch_size or 1
    image_size = args.image_size or [800, 800]

    print("Creating model")
    model = None
    if args.backbone=="resnet50_fpn":
        model = retinanet_resnet50_fpn(num_classes=args.num_classes,
                                       pretrained=None,
                                       image_size=image_size)
    elif args.backbone=="resnext50_32x4d_fpn":
        model = retinanet_resnext50_32x4d_fpn(num_classes=args.num_classes,
                                              pretrained=None,
                                              image_size=image_size)
    device = torch.device(args.device)
    model.to(device)

    print("Loading model")
    checkpoint = torch.load(args.input)
    model.load_state_dict(checkpoint['model'])

    print("Creating input tensor")
    rand = torch.randn(batch_size, 3, image_size[0], image_size[1],
                       device=device,
                       requires_grad=False,
                       dtype=torch.float)
    inputs = torch.autograd.Variable(rand)
    # Output dynamic axes
    dynamic_axes = {
        'boxes': {0 : 'num_detections'},
        'scores': {0 : 'num_detections'},
        'labels': {0 : 'num_detections'},
    }
    # Input dynamic axes
    if (args.batch_size is None) or (args.image_size is None):
        dynamic_axes['images'] = {}
        if args.batch_size is None:
            dynamic_axes['images'][0]: 'batch_size'
        if args.image_size is None:
            dynamic_axes['images'][2] = 'width'
            dynamic_axes['images'][3] = 'height'


    print("Exporting the model")
    model.eval()
    torch.onnx.export(model,
                      inputs,
                      args.output,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=False,
                      input_names=['images'],
                      output_names=['boxes', 'scores', 'labels'],
                      dynamic_axes=dynamic_axes)


if __name__ == "__main__":
    args = parse_args()
    main(args)
