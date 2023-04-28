import torch
import torch.nn as nn
from torchvision import models
from scc4onnx import order_conversion
import onnx
from onnx_tf.backend import prepare
import subprocess

#
# PyTorch Model Definition
#
class MobilenetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(MobilenetV2, self).__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V2')
        self.model.classifier[1]= nn.Linear(1280, num_classes, bias=False)

    def forward(self, x):
        return self.model(x)

#
# Load Pytorch Model
#
model = MobilenetV2(num_classes=4)
model.load_state_dict(torch.load('model_state.pt', map_location='cpu'))
model.eval()

#
# Pytorch Model -> ONNX Model
#
dummy_input = torch.randn(1, 3, 232, 232, requires_grad=True)
torch.onnx.export(
    model,
    dummy_input,
    'telomere_model.onnx'
)

#
# ONNX Model -> Tensorflow Model
#
tf_model_dir = "telomere_model_tf"
onnx_model = onnx.load('telomere_model.onnx')
input_name = onnx_model.graph.input[0].name
onnx_model = order_conversion(
    onnx_graph=onnx_model,
    input_op_names_and_order_dims={f"{input_name}": [0,2,3,1]},
    non_verbose=True
)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_dir)

#
# Tensorflow Model -> Tensorflow.js Model
#
tfjs_model_dir = "telomere_model_tfjs"
tfjs_convert_command = [
    'tensorflowjs_converter',
    '--input_format=tf_saved_model',
    '--output_format=tfjs_graph_model',
    '--signature_name=serving_default',
    '--saved_model_tags=serve',
    tf_model_dir,
    tfjs_model_dir
]
subprocess.run(tfjs_convert_command)
