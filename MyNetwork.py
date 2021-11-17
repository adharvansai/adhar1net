import torch
import torch.nn as nn
from torch.functional import Tensor
import torch.nn.functional as F

class batch_norm_relu_layer(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        self.nf = num_features
        self.eps_value = eps
        self.mom = momentum

        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum)

    def forward(self, inputs: Tensor) -> Tensor:

        out = F.relu(self.batch_norm(inputs))
        return out

class Adhar1_block(nn.Module):
    def __init__(self, input_ch,num_of_conv):
        super(Adhar1_block, self).__init__()

        self.num_of_conv = num_of_conv
        self.layers = []
        self.bn = []
        self.layers.append(nn.Conv2d(in_channels = input_ch, out_channels = 32, kernel_size = 3, stride = 1, padding = 1))
        self.bn.append(batch_norm_relu_layer(input_ch))
        for i in range(1,num_of_conv):
            self.layers.append(nn.Conv2d(in_channels = int(i*32), out_channels = 32, kernel_size = 3, stride = 1, padding = 1))
            self.bn.append(batch_norm_relu_layer(i*32))

    def forward(self, x):

        conv_out = []

        bn_out = self.bn[0](x)
        conv_out.append(F.relu(self.layers[0](bn_out)))
        bn_out = self.bn[1](conv_out[0])
        conv_out.append(F.relu(self.layers[1](bn_out)))
        for i in range(2,self.num_of_conv):
            curr_conv = torch.cat(conv_out[:i],1)
            bn_out = self.bn[i](curr_conv)
            conv_out.append(self.layers[i](bn_out))

        out = F.relu(torch.cat(conv_out,1))

        return out

class Trasition_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Trasition_layer, self).__init__()

        self.bn = batch_norm_relu_layer(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)
        out = self.avg_pool(bn_out)
        return out

class MyNetwork(nn.Module):
    def __init__(self, configs):
        super(MyNetwork, self).__init__()
        self.configs = configs
        self.num_classes = configs['num_classes']
        self.num_of_blocks = configs['network_size']
        self.start_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False)
        self.star_bn = batch_norm_relu_layer(64)

        self.block1 = self._make_block(64)
        self.block2 = self._make_block(128)
        self.block3 = self._make_block(128)

        self.transition1 = self._make_layer(self.num_of_blocks*32,128)
        self.transition2 = self._make_layer(self.num_of_blocks*32, 128)
        self.transition3 = self._make_layer(self.num_of_blocks*32, 64)

        self.bn = batch_norm_relu_layer(64)
        self.pre_classifier = nn.Linear(64 * 4 * 4, 512)
        self.classifier = nn.Linear(512, self.num_classes)

    def _make_block(self,in_ch):
        layers = []
        layers.append(Adhar1_block(in_ch,self.num_of_blocks))
        return nn.Sequential(*layers)
    def _make_layer(self,in_ch,out_ch):
        components = []
        components.append(Trasition_layer(in_ch,out_ch))
        return nn.Sequential(*components)

    def forward(self,x):
        out = self.start_layer(x)
        out = self.star_bn(out)

        out = self.block1(out)
        out = self.transition1(out)

        out = self.block2(out)
        out = self.transition2(out)

        out = self.block3(out)
        out = self.transition3(out)

        out = self.bn(out)
        out = out.view(-1, 64 * 4 * 4)

        out = self.pre_classifier(out)
        out = self.classifier(out)
        return out










