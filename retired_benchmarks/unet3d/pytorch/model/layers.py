import torch
import torch.nn as nn

from mlperf_logging.mllog import constants
from runtime.logging import mllog_event

activations = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.01),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=1),
    "none": nn.Identity(),
}

normalizations = {
    "instancenorm": lambda n, _: nn.InstanceNorm3d(n, affine=True),
    "batchnorm": lambda n, _: nn.BatchNorm3d(n),
    "syncbatchnorm": lambda n, _: nn.SyncBatchNorm(n),
    "none": lambda _, __: nn.Identity(),
}

convolutions = {"transpose": nn.ConvTranspose3d, "regular": nn.Conv3d}


def _normalization(norm_type, num_features, num_groups=16):
    if norm_type in normalizations:
        return normalizations[norm_type](num_features, num_groups)
    raise ValueError(f"Unknown normalization {norm_type}")


def _activation(activation):
    if activation in activations:
        return activations[activation]
    raise ValueError(f"Unknown activation {activation}")


def conv_block_factory(in_channels, out_channels,
                       kernel_size=3, stride=1, padding=1,
                       conv_type="regular", name="",
                       norm_type="instancenorm", activation="relu"):
    suffix = "_conv" if conv_type == "regular" else "_deconv"
    conv = convolutions[conv_type]
    conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=norm_type == "none")

    mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=name + suffix))
    normalization = _normalization(norm_type, out_channels)
    if norm_type == "instancenorm":
        mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=name + "_instancenorm"))
    activation = _activation(activation)

    return nn.Sequential(conv, normalization, activation)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, activation, index):
        super(DownsampleBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, stride=2, name=f"down{index}_block_0",
                                        norm_type=normalization, activation=activation)
        self.conv2 = conv_block_factory(out_channels, out_channels, name=f"down{index}_block_1",
                                        norm_type=normalization, activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, activation, index):
        super(UpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_conv = conv_block_factory(in_channels, out_channels,
                                                kernel_size=2, stride=2, padding=0, name=f"up{index}",
                                                conv_type="transpose", norm_type="none", activation="none")
        self.conv1 = conv_block_factory(2 * out_channels, out_channels, name=f"up{index}_block_0",
                                        norm_type=normalization, activation=activation)
        self.conv2 = conv_block_factory(out_channels, out_channels, name=f"up{index}_block_1",
                                        norm_type=normalization, activation=activation)

    def forward(self, x, skip):
        x = self.upsample_conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization, activation):
        super(InputBlock, self).__init__()
        self.conv1 = conv_block_factory(in_channels, out_channels, name="input_block_0",
                                        norm_type=normalization, activation=activation)
        self.conv2 = conv_block_factory(out_channels, out_channels, name="input_block_1",
                                        norm_type=normalization, activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, in_channels, n_class):
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, n_class, kernel_size=1, stride=1, padding=0, bias=True)
        mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=f"output_conv"))

    def forward(self, x):
        return self.conv(x)
