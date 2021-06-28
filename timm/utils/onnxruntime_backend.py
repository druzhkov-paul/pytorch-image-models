import onnx
import onnxruntime
from onnx import helper, shape_inference
from onnx.utils import polish_model

class ModelONNXRuntime:

    def __init__(self, model_file_path, cfg=None, classes=None):
        self.device = onnxruntime.get_device()
        self.model = onnx.load(model_file_path)
        # self.model = polish_model(self.model)
        self.classes = classes

        self.sess_options = onnxruntime.SessionOptions()
        # self.sess_options.enable_profiling = False

        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)
        self.input_names = []
        self.output_names = []
        for input in self.session.get_inputs():
            self.input_names.append(input.name)
        for output in self.session.get_outputs():
            self.output_names.append(output.name)

    def add_output(self, output_ids):
        if not isinstance(output_ids, (tuple, list, set)):
            output_ids = [
                output_ids,
            ]

        inferred_model = shape_inference.infer_shapes(self.model)
        all_blobs_info = {
            value_info.name: value_info
            for value_info in inferred_model.graph.value_info
        }

        extra_outputs = []
        for output_id in output_ids:
            value_info = all_blobs_info.get(output_id, None)
            if value_info is None:
                print('WARNING! No blob with name {}'.format(output_id))
                extra_outputs.append(
                    helper.make_empty_tensor_value_info(output_id))
            else:
                extra_outputs.append(value_info)

        self.model.graph.output.extend(extra_outputs)
        self.output_names.extend(output_ids)
        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            if len(self.input_names) == 1 and not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = dict(zip(self.input_names, inputs))
        return inputs

    def __call__(self, inputs, *args, **kwargs):
        inputs = self.unify_inputs(inputs)
        outputs = self.session.run(None, inputs, *args, **kwargs)
        outputs = dict(zip(self.output_names, outputs))
        return outputs
