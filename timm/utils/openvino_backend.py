import logging
import numpy as np
import os.path as osp

from openvino.inference_engine import IECore


class XDict:
    def __init__(self, d, net):
        self.d = d
        self.net = net

    def __getitem__(self, name: str) -> Any:
        return self.net.get(self.d, name)

    def __getattribute__(self, name: str) -> Any:
        return getattr(self.d, name)


class ModelOpenVINO:
    def __init__(self, model_path, ie=None, device='CPU'):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')

        self.ie = IECore() if ie is None else ie
        bin_path = osp.splitext(model_path)[0] + '.bin'
        self.net = self.ie.read_network(model_path, bin_path)

        self.device = None
        self.exec_net = None
        self.to(device)

    def to(self, device):
        if self.device != device:
            self.device = device
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)
        return self

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def reshape(self, inputs=None, input_shapes=None):
        assert (inputs is None) != (input_shapes is None)
        if input_shapes is None:
            input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            self.logger.info(f'reshape net to {input_shapes}')
            self.net.reshape(input_shapes)
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)

    def get(self, outputs, name):
        try:
            key = self.net.get_ov_name_for_tensor(name)
            assert key in outputs, f'"{key}" is not a valid output identifier'
        except KeyError:
            if name not in outputs:
                raise KeyError(f'Failed to identify output "{name}"')
            key = name
        return outputs[key]


    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        self.reshape(inputs=inputs)
        outputs = self.exec_net.infer(inputs)
        return XDict(outputs, self)
