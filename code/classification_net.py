import torch
import torch.nn as nn

class Fusion_vgg_resnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fusion_layer = nn.Linear(3000,512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc1  = nn.Linear(512,256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256,4)
        # self.log = nn.LogSoftmax(dim=1)

        self._initialize_weights()


    def forward(self,fusion_input):
        # fusion_input = torch.cat((x_resnet,x_vgg,se_out),dim=1)
        out = self.fusion_layer(fusion_input)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        # out = self.log(out)

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
