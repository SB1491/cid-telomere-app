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
        self.telo_append = nn.Sequential(
            nn.Linear(6, 28),
            nn.Tanh()
        )
        self.telo_neural = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes, bias=False)
        )
    
    def forward(self, x, metadata):
        A = self.MV2A(x)
        B = self.MV2B(x)
        C = self.MV2C(x)
        y = torch.cat([A, B, C], dim=1)
        y = self.telo_append(y)
        y = torch.cat([y, metadata], dim=1)
        return self.telo_neural(y)

class SuperTeloMobilenetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(SuperTeloMobilenetV2, self).__init__()
        self.TMV1 = TeloMobilenetV2(num_classes=num_classes)
        self.TMV2 = TeloMobilenetV2(num_classes=num_classes)
        self.TMV3 = TeloMobilenetV2(num_classes=num_classes)
        self.TMV4 = TeloMobilenetV2(num_classes=num_classes)
        self.TMV5 = TeloMobilenetV2(num_classes=num_classes)
        self.TMV6 = TeloMobilenetV2(num_classes=num_classes)

    def forward(self, x, metadata):
        out1 = self.TMV1(x, metadata)
        out2 = self.TMV2(x, metadata)
        out3 = self.TMV3(x, metadata)
        out4 = self.TMV4(x, metadata)
        out5 = self.TMV5(x, metadata)
        out6 = self.TMV6(x, metadata)
        return torch.cat([out1, out2, out3, out4, out5, out6], dim=1)


#
# Load Pytorch Model
#
model = SuperTeloMobilenetV2()
model.TMV1.load_state_dict(torch.load('model1_weight.pt', map_location='cpu'))
model.TMV2.load_state_dict(torch.load('model2_weight.pt', map_location='cpu'))
model.TMV3.load_state_dict(torch.load('model3_weight.pt', map_location='cpu'))
model.TMV4.load_state_dict(torch.load('model4_weight.pt', map_location='cpu'))
model.TMV5.load_state_dict(torch.load('model5_weight.pt', map_location='cpu'))
model.TMV6.load_state_dict(torch.load('model6_weight.pt', map_location='cpu'))
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
