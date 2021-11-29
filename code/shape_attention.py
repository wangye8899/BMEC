from typing import ForwardRef
import torch 
import torch.nn as nn


class shape_attentionNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(1000,256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256,1000)
        # self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def forward(self,x):

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        # out = self.relu2(out)
        out = self.sigmoid(out)

        return out
        
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class cross_attention(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(3000,1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024,3000)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out
    




    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)