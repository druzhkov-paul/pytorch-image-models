from functools import wraps

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op, get_registered_op, is_registered_op


def py_symbolic(op_name=None, namespace='timm_custom', adapter=None):
    """
    The py_symbolic decorator allows associating a function with a custom symbolic function
    that defines its representation in a computational graph.
    A symbolic function cannot receive a collection of tensors as arguments.
    If your custom function takes a collection of tensors as arguments,
    then you need to implement an argument converter (adapter) from the collection
    and pass it to the decorator.
    Args:
        op_name (str): Operation name, must match the registered operation name.
        namespace (str): Namespace for this operation.
        adapter (function): Function for converting arguments.
    Adapter conventions:
        1. The adapter must have the same signature as the wrapped function.
        2. The values, returned by the adapter, must match the called symbolic function.
        3. Return value order:
            tensor values (collections are not supported)
            constant parameters (can be passed using a dictionary)
    Usage example:
        1. Implement a custom operation. For example 'custom_op'.
        2. Implement a symbolic function to represent the custom_op in
            a computation graph. For example 'custom_op_symbolic'.
        3. Register the operation before export:
            register_op('custom_op_name', custom_op_symbolic, namespace, opset)
        4. Decorate the custom operation:
            @py_symbolic(op_name='custom_op_name')
            def custom_op(...):
        5. If you need to convert custom function arguments to symbolic function arguments,
            you can implement a converter and pass it to the decorator:
            @py_symbolic(op_name='custom_op_name', adapter=converter)
    """
    def decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            name = op_name if op_name is not None else func.__name__
            opset = sym_help._export_onnx_opset_version

            if is_registered_op(name, namespace, opset):

                class XFunction(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, *xargs):
                        return func(*args, **kwargs)

                    @staticmethod
                    def symbolic(g, *xargs):
                        symb = get_registered_op(name, namespace, opset)
                        if adapter is not None:
                            return symb(g, *xargs, **adapter_kwargs)
                        return symb(g, *xargs)

                if adapter is not None:
                    adapter_args, adapter_kwargs = adapter(*args, **kwargs)
                    return XFunction.apply(*adapter_args)
                return XFunction.apply(*args)
            else:
                return func(*args, **kwargs)
        return wrapped_function
    return decorator
