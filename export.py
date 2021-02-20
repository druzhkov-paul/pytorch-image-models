import argparse

import numpy as np
import timm
import torch

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


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


if __name__ == '__main__':
    args = parse_args()

    model = timm.create_model(args.model_name, pretrained=False, exportable=True)
    # model.load_state_dict(torch.load('model_best.pth.tar', map_location=device))
    model.eval()
    input_shape = (1, ) + model.default_cfg['test_input_size']
    print(input_shape)
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_file,
        export_params=True,
        keep_initializers_as_inputs=True,
        verbose=True,
        opset_version=args.opset_version,
        input_names=['image'],
        output_names=['probs'],
        strip_doc_string=False,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        # dynamic_axes = {
        #     "image": {
        #         2: "height",
        #         3: "width"
        #     }
        # }
    )
