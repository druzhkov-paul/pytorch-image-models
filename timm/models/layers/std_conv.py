""" Convolution with Weight Standardization (StdConv and ScaledStdConv)

StdConv:
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
Code: https://github.com/joe-siyuan-qiao/WeightStandardization

ScaledStdConv:
Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets`
    - https://arxiv.org/abs/2101.08692
Official Deepmind JAX code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Hacked together by / copyright Ross Wightman, 2021.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .padding import get_padding, get_padding_value, pad_same
from ...utils.export import py_symbolic


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-6):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class StdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. TF compatible SAME padding. Used for ViT Hybrid model.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding='SAME',
            dilation=1, groups=1, bias=False, eps=1e-6):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.same_pad = is_dynamic
        self.eps = eps

    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        weight = F.batch_norm(
            self.weight.view(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ScaledStdConv2d(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps
        self.legacy_mode = False

    def get_weight(self):
        if self.legacy_mode:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.gain * self.scale * (self.weight - mean) / (std + self.eps)
        else:
            weight = F.batch_norm(
                self.weight.view(1, self.out_channels, -1), None, None,
                weight=(self.gain * self.scale).view(-1),
                training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        return weight

    def forward(self, x):
        if self.eval:
            weight = self.ready_weights
        else:
            weight = self.get_weight()

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ScaledStdConv2dSame(nn.Conv2d):
    """Conv2d layer with Scaled Weight Standardization and Tensorflow-like SAME padding support

    Paper: `Characterizing signal propagation to close the performance gap in unnormalized ResNets` -
        https://arxiv.org/abs/2101.08692

    NOTE: the operations used in this impl differ slightly from the DeepMind Haiku impl. The impact is minor.
    """

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding='SAME',
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.same_pad = is_dynamic
        self.eps = eps

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.ready_weights = self.get_weight().detach()
        else:
            self.ready_weights = None

    def get_weight(self):
        if self.legacy_mode:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.gain * self.scale * (self.weight - mean) / (std + self.eps)
        else:
            weight = F.batch_norm(self.weight.view(1, self.out_channels, -1), None, None,
                weight=(self.gain * self.scale).view(-1),
                training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        return weight

    def forward(self, x):
        # import torch.jit
        # from torch.jit import _disable_tracing

        # if torch.onnx.is_in_onnx_export():
        #     with _disable_tracing():
        #         weights = self.get_weight()
        #         weights = weights.detach()
        # else:
        #     weights = self.get_weight()

        if self.eval:
            weights = self.ready_weights
        else:
            weights = self.get_weight()

        if self.same_pad:
            return conv_same_pad(x, weights, self.bias, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(x, weights, self.bias, self.stride, self.padding, self.dilation, self.groups)


@py_symbolic()
def conv_same_pad(x, weights, bias, kernel_size, stride, padding, dilation, groups):
    # print('conv_same_pad')
    x = pad_same(x, kernel_size, stride, dilation)
    return F.conv2d(x, weights, bias, stride, padding, dilation, groups)


class _disable_tracing(object):
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None
