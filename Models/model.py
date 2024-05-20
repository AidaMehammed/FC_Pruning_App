import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, quantize=False):
        super().__init__()
        self.quantize = quantize

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # FloatFunction()
        self.skip_add = nn.quantized.FloatFunctional()
        # for inputs
        self.quant = torch.quantization.QuantStub()
        # for outputs
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        if self.quantize:
          x = self.quant(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.quantize:
          x = self.dequant(x)

        return x

    def get_weights(self):
      return [param.data for param in self.parameters()]


    def set_weights(self, weights):
      for param, weight in zip(self.parameters(), weights):
            param.data = weight
