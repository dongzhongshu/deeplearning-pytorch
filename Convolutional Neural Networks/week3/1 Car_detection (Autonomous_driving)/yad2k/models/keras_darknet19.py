"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial

# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import LeakyReLU
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2
import torch
from torch.nn import Conv2d, MaxPool2d, LeakyReLU, BatchNorm2d, Module
from torch.optim.optimizer import l2

from ..utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2d, padding='same')


# @functools.wraps(Conv2d)
# def DarknetConv2D(*args, **kwargs):
#     """Wrapper to set Darknet weight regularizer for Convolution2D."""
#     darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
#     darknet_conv_kwargs.update(kwargs)
#     return _DarknetConv2D(*args, **darknet_conv_kwargs)
#
#
# def DarknetConv2D_BN_Leaky(*args, **kwargs):
#     """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
#     no_bias_kwargs = {'use_bias': False}
#     no_bias_kwargs.update(kwargs)
#     return compose(
#         DarknetConv2D(*args, **no_bias_kwargs),
#         # BatchNormalization(),
#         BatchNorm2d(),
#         LeakyReLU(0.1))


class DarknetConv2D_BN_Leaky(Module):
    def __init__(self, inchannels, outchannels, **args):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv = Conv2d(inchannels, outchannels, **args)
        self.batchnormal = BatchNorm2d(outchannels)
        self.relu = LeakyReLU(0.1)
    def forward(self, X):
        x = self.conv(X)
        x = self.batchnormal(x)
        x = self.relu(x)
        return x

class bottleneck_block(Module):
    def __init__(self, inchannels, outer_filters, bottleneck_filters):
        super(bottleneck_block).__init__()
        self.DarknetConv2D1 = DarknetConv2D_BN_Leaky(inchannels, outer_filters, kernel_size=(3, 3))
        self.DarknetConv2D2 = DarknetConv2D_BN_Leaky(outer_filters, bottleneck_filters, kernel_size=(1, 1))
        self.DarknetConv2D3 = DarknetConv2D_BN_Leaky(bottleneck_filters, outer_filters, kernel_size=(3, 3))
    def forward(self, X):
        x = self.DarknetConv2D1(X)
        x = self.DarknetConv2D2(x)
        x = self.DarknetConv2D3(x)
        return x
# def bottleneck_block(outer_filters, bottleneck_filters):
#     """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""
#     return compose(
#         DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
#         DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
#         DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))



class bottleneck_x2_block(Module):
    def __init__(self, inchannels, outer_filters, bottleneck_filters):
        super(bottleneck_x2_block, self).__init__()
        self.bottleneck = bottleneck_x2_block(inchannels, outer_filters, bottleneck_filters)
        self.DarknetConv2D1 = DarknetConv2D_BN_Leaky(outer_filters, bottleneck_filters, kernel_size=(1, 1))
        self.DarknetConv2D2 = DarknetConv2D_BN_Leaky(bottleneck_filters, outer_filters, kernel_size=(3, 3))
    def forward(self, X):
        x = self.bottleneck(X)
        x = self.DarknetConv2D1(x)
        x = self.DarknetConv2D2(x)
        return x


# def bottleneck_x2_block(outer_filters, bottleneck_filters):
#     """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""
#     return compose(
#         bottleneck_block(outer_filters, bottleneck_filters),
#         DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
#         DarknetConv2D_BN_Leaky(outer_filters, (3, 3)))


class darknet_body(Module):
    def __init__(self, inchannels):
        self.DarknetConv2D1 = DarknetConv2D_BN_Leaky(inchannels, 32, kernel_size=(3, 3))
        self.maxpool2d = MaxPool2d(2)
        self.DarknetConv2D2 = DarknetConv2D_BN_Leaky(32, 64, kernel_size=(3, 3))
        self.bottleneck1 = bottleneck_block(64, 128, 64)
        self.bottleneck2 = bottleneck_block(64, 256, 128)
        self.bottleneck_x21 = bottleneck_x2_block(128, 512, 256)
        self.bottleneck_x22 = bottleneck_x2_block(256, 1024, 512)
        self.fc = Conv2d(512, 1000, kernel_size=(1, 1))
    def forward(self, X):
        x = self.DarknetConv2D1(X)
        x = self.maxpool2d(x)
        x = self.DarknetConv2D2(x)
        x = self.maxpool2d(x)
        x = self.bottleneck1(x)
        x = self.maxpool2d(x)
        x = self.bottleneck2(x)
        x = self.maxpool2d(x)
        x = self.bottleneck_x21(x)
        x = self.maxpool2d(x)
        x = self.bottleneck_x22(x)
        x = self.fc(x)
        x = torch.nn.Flatten()(x)
        return x


# def darknet_body():
#     """Generate first 18 conv layers of Darknet-19."""
#     return compose(
#         DarknetConv2D_BN_Leaky(32, (3, 3)),
#         MaxPooling2D(),
#         DarknetConv2D_BN_Leaky(64, (3, 3)),
#         MaxPooling2D(),
#         bottleneck_block(128, 64),
#         MaxPooling2D(),
#         bottleneck_block(256, 128),
#         MaxPooling2D(),
#         bottleneck_x2_block(512, 256),
#         MaxPooling2D(),
#         bottleneck_x2_block(1024, 512))


def darknet19(inputs):
    """Generate Darknet-19 model for Imagenet classification."""
    model = darknet_body(3)
    x = model(inputs)
    return x
    # body = darknet_body()(inputs)
    # logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)
    # return Model(inputs, logits)
