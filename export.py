import argparse
from functools import wraps
from sys import maxsize

import numpy as np
import timm
import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op, get_registered_op, is_registered_op
from torch.onnx.symbolic_opset9 import _convolution

try:
    import onnx
    import onnxruntime as rt
    from onnxoptimizer import optimize
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


@parse_args('v', 'is', 'is')
def roll(g, self, shifts, dims):
    assert len(shifts) == len(dims)

    result = self
    for i in range(len(shifts)):
        shapes = []
        shape = sym_help._slice_helper(g,
                                       result,
                                       axes=[dims[i]],
                                       starts=[-shifts[i]],
                                       ends=[maxsize])
        shapes.append(shape)
        shape = sym_help._slice_helper(g,
                                       result,
                                       axes=[dims[i]],
                                       starts=[0],
                                       ends=[-shifts[i]])
        shapes.append(shape)
        result = g.op("Concat", *shapes, axis_i=dims[i])

    return result


@parse_args('v', 'v', 'v', 'v', 'none')
def addcmul_symbolic(g, self, tensor1, tensor2, value=1, out=None):
    from torch.onnx.symbolic_opset9 import add, mul

    if out is not None:
        sym_help._unimplemented("addcmul", "Out parameter is not supported for addcmul")

    x = mul(g, tensor1, tensor2)
    value = sym_help._maybe_get_scalar(value)
    if sym_help._scalar(value) != 1:
        value = sym_help._if_scalar_type_as(g, value, x)
        if not sym_help._is_value(value):
            value = g.op(
                "Constant", value_t=torch.tensor(value, dtype=torch.float32))
        x = mul(g, x, value)
    return add(g, self, x)


@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'is', 'i')
def conv_same_pad_symbolic(g, x, weights, bias, kernel_size, stride, padding, dilation, groups):
    print('conv_same_pad_symbolic')
    weight_size = weights.type().sizes()

    args = [x, weights]
    # ONNX only supports 1D bias
    if not sym_help._is_none(bias) and bias.type().dim() == 1:
        args.append(bias)

    kwargs = {"auto_pad_s": "SAME_UPPER",
              "kernel_shape_i": weight_size[2:],
              "strides_i": stride,
              # NB: ONNX supports asymmetric padding, whereas PyTorch supports only
              # symmetric padding
              # "pads_i": padding + padding,
              "dilations_i": dilation,
              "group_i": groups}

    n = g.op("Conv", *args, **kwargs)

    if not sym_help._is_none(bias) and bias.type().dim() != 1:
        return g.op("Add", n, bias)
    else:
        return n

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model_name', help='')
    # parser.add_argument('-ckpt', '--checkpoint', default=None, help='checkpoint file')
    # parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument('--dyn-batch', action='store_true')
    parser.add_argument('--dyn-res', action='store_true')
    parser.add_argument('--clean', action='store_true')
    # parser.add_argument(
    #     '--shape',
    #     type=int,
    #     nargs='+',
    #     default=[1, 3, 256, 192],
    #     help='input size')
    args = parser.parse_args()
    return args

# from torch.onnx.symbolic_helper import parse_args

# def std_mean_symbolic(g, input, dim, unbiased=True, keepdim=False):
#     mean = g.op('ReduceMean', input, axes_i=dim, keepdims_i=int(keepdim))
#     std = g.op('Sqrt', g.op('ReduceSumSquare', input - mean, axes_i=dim, keepdims_i=int(keepdim)))
#     return mean, std


def optimize_onnx_graph(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)

    onnx_model = optimize(onnx_model, ['extract_constant_to_initializer',
                                       'eliminate_unused_initializer'])

    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(onnx_model, onnx_model_path)


if __name__ == '__main__':
    args = parse_args()

    torch.onnx.symbolic_registry.register_op('conv_same_pad', conv_same_pad_symbolic, 'timm_custom', args.opset_version)
    torch.onnx.symbolic_registry.register_op('roll', roll, '', args.opset_version)
    torch.onnx.symbolic_registry.register_op('addcmul', addcmul_symbolic, '', args.opset_version)

    model = timm.create_model(args.model_name, pretrained=True, exportable=True, pretrained_strict=False)
    # model.load_state_dict(torch.load('model_best.pth.tar', map_location=device))
    model.eval()
    print(model.default_cfg)
    # input_shape = (1, ) + model.default_cfg['test_input_size']
    input_shape = (1, ) + model.default_cfg['input_size']
    print(input_shape)
    dummy_input = torch.randn(*input_shape)
    # print('Run inference without tracing')
    all_outputs = model(dummy_input)
    print(f'Network has {len(all_outputs)} output(s)')

    try:
        output_names = list(all_outputs.keys())
    except AttributeError:
        if len(all_outputs) == 1:
            output_names = ['probs']
        else:
            output_names = [f'out_{i}' for i, _ in enumerate(all_outputs)]

    dynamic_axes = {}
    if args.dyn_batch:
        dynamic_axes.setdefault('image', {})[0] = "batch_size"
        for k in output_names:
            dynamic_axes.setdefault(k, {})[0] = "batch_size"
    if args.dyn_res:
        dynamic_axes.setdefault('image', {})[2] = "height"
        dynamic_axes.setdefault('image', {})[3] = "width"
    # print('Start export')
    torch.onnx.export(
        model,
        dummy_input,
        args.output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=False,
        opset_version=args.opset_version,
        input_names=['image'],
        # output_names=['probs'],
        output_names=output_names,
        strip_doc_string=args.clean,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        dynamic_axes=dynamic_axes
    )
    optimize_onnx_graph(args.output_file)
