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
        self.model = models.mobilenet_v2()
        self.model.classifier[1]= nn.Linear(1280, num_classes, bias=False)

    def forward(self, x):
        return self.model(x)

class TeloMobilenetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(TeloMobilenetV2, self).__init__()
        self.MV2A = MobilenetV2(2)
        self.MV2B = MobilenetV2(2)
        self.MV2C = MobilenetV2(2)
        self.telo_neural= nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, num_classes, bias=False)
        )
    
    def forward(self, x, metadata):
        A = self.MV2A(x)
        B = self.MV2B(x)
        C = self.MV2C(x)
        y = torch.cat([A, B, C, metadata], dim=1)
        return self.telo_neural(y)


#
# Load Pytorch Model
#
model = TeloMobilenetV2()
model.load_state_dict(torch.load('model_state.pt', map_location='cpu'))
model.eval()

#
# Pytorch Model -> ONNX Model
#
dummy_input = torch.randn(1, 3, 232, 232, requires_grad=True)
dummy_metadata = torch.Tensor([[0, 0.5, 0.5, 0]])
torch.onnx.export(
    model,
    (dummy_input, dummy_metadata),
    'telomere_model.onnx'
)

#
# ONNX Model -> Tensorflow Model
#
tf_model_dir = "telomere_model_tf"
onnx_model = onnx.load('telomere_model.onnx')
input_names = [onnx_model.graph.input[i].name for i in range(2)]
input_dim_orders = [[0, 2, 3, 1], [0, 1]]
onnx_model = order_conversion(
    onnx_graph=onnx_model,
    input_op_names_and_order_dims={
        f"{input_names[i]}": input_dim_orders[i] for i in range(2)
    },
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
