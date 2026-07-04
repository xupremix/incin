import torch
import torch.nn as nn
import os

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc(x))

model = BasicModel()
x = torch.randn(1, 10)
os.makedirs("/home/xupremix/kindle/crates/kindle/tests", exist_ok=True)
torch.onnx.export(model, x, "/home/xupremix/kindle/crates/kindle/tests/basic_model.onnx", input_names=["input"], output_names=["output"])
