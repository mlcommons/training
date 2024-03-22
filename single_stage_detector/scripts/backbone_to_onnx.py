#!/usr/bin/env python3
import argparse

import torch
import torch.onnx
from torchvision.ops import misc as misc_nn_ops
from torch.autograd import Variable

from model.resnet import resnet50, resnext50_32x4d


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Convert RetinaNet backbone to onnx format', add_help=add_help)

    parser.add_argument('--backbone', default='resnext50_32x4d', choices=['resnet50', 'resnext50_32x4d'],
                        help='The model backbone')
    parser.add_argument('--output', default=None, help='output onnx file')
    parser.add_argument('--image-size', default=None, nargs=2, type=int,
                        help='Image size for training. If not set then will be dynamic')
    parser.add_argument('--batch-size', default=None, type=int,
                        help='input batch size. if not set then will be dynamic')
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()

    args.output = args.output or (args.backbone+'.onnx')
    return args

def main(args):
    batch_size = args.batch_size or 1
    image_size = args.image_size or [800, 800]

    print("Loading model")
    model = None
    if args.backbone=="resnet50":
        model = resnet50(pretrained=True,
                         norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif args.backbone=="resnext50_32x4d":
        model = resnext50_32x4d(pretrained=True,
                                norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    device = torch.device(args.device)
    model.to(device)

    print("Creating input tensor")
    rand = torch.randn(batch_size, 3, image_size[0], image_size[1],
                       device=device,
                       requires_grad=False,
                       dtype=torch.float)
    inputs = torch.autograd.Variable(rand)
    dynamic_axes = {}
    # Input dynamic axes
    if (args.batch_size is None) or (args.image_size is None):
        dynamic_axes['images'] = {}
        if args.batch_size is None:
            dynamic_axes['images'][0]: 'batch_size'
        if args.image_size is None:
            dynamic_axes['images'][2] = 'width'
            dynamic_axes['images'][3] = 'height'


    print("Exporting the model")
    torch.onnx.export(model,
                      inputs,
                      args.output,
                      export_params=True,
                      opset_version=13,
                      input_names=['images'],
                      dynamic_axes=dynamic_axes)


if __name__ == "__main__":
    args = parse_args()
    main(args)
