import torch
from torchvision import models
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
from symbols.common_symbols import Conv, MaxPool, DecConvBn, FC, FCRelu, FCSigmoid, Add

class MobilenetV3(torch.nn.Module):
    def __init__(self, pretrained=True, scale=2, feat_dim=64) -> None:
        super().__init__()
        self.scale = scale
        self.pretrained = pretrained
        
        backbone = models.mobilenet_v3_small()
        self.model = torch.nn.Sequential(
            backbone.features, backbone.avgpool
        )
        if self.pretrained:
            pretrained_dict = self.model.state_dict()

        self.fc = Linear(576, feat_dim*scale)
        self.fc1 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig1 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # pretty
        self.fc2 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig2 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # blur
        self.fc3 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig3 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # glass
        self.fc4 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig4 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # makeup
        self.fc5 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig5 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # gender
        self.fc6 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig6 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # mouthopen
        self.fc7 = FCSigmoid(c_in=int(64 * self.scale), c_out=int(32 * self.scale))
        self.sig7 = FCSigmoid(c_in=int(32 * self.scale), c_out=1)        # smile

        self._initialize_weights(pretrained_dict)
        del pretrained_dict

    def _initialize_weights(self, pretrained_dict):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
        
        if self.pretrained:
            self.model.load_state_dict(pretrained_dict)
        

    def forward(self, x) -> torch.tensor:
        x = self.model(x).squeeze()
        # print('xxxxxxxxxxxxx',x.shape)
        x = self.fc(x)
        pretty = self.sig1(self.fc1(x))
        blur = self.sig2(self.fc2(x))
        glass = self.sig3(self.fc3(x))
        makeup = self.sig4(self.fc4(x))
        gender = self.sig5(self.fc5(x))
        mouthopen = self.sig6(self.fc6(x))
        smile = self.sig7(self.fc7(x))
        return pretty, blur, glass, makeup, gender, mouthopen, smile
    

def get_mobilenetv3(pretrained=True, scale=2, feat_dim=64):
    return MobilenetV3(pretrained, scale, feat_dim)
