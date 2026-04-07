import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):#resnet18、resnet34
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):#resnet50、resnet101、resnet152
    ''' Standard bottleneck block
    input  = inplanes * H * W
    middle =   planes * H/stride * W/stride
    output = 4*planes * H/stride * W/stride
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):#stride可能会变
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)#dilation不是很懂
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




def reset_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ResNet(nn.Module):
    '''示例：如果用DataParallel包裹的话
    module.conv1.weight
    module.bn1.weight
    module.bn1.bias
    module.bn1.running_mean
    module.bn1.running_var
    module.bn1.num_batches_tracked
    module.layer1.0.conv1.weight #.0应该是表示第0个block
    module.layer1.0.bn1.weight
    
    '''
    """ A standard ResNet.
    """
    def __init__(self, block, layers, fc_out, model_name, self_similarity_radius=None, self_similarity_version=2):
        nn.Module.__init__(self)
        self.model_name = model_name

        # default values for a network pre-trained on imagenet
        self.rgb_means = [0.485, 0.456, 0.406]
        self.rgb_stds  = [0.229, 0.224, 0.225]
        self.input_size = (3, 224, 224)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)#(B,3,224,224)->(B,64,112,112)
        self.bn1 = nn.BatchNorm2d(64)#固定通道这一维度，对B*H*W这些元素加和求平均、求方差
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#(B,64,112,112)->(B,64,56,56)
        self.layer1 = self._make_layer(block, 64, layers[0], self_similarity_radius=self_similarity_radius, self_similarity_version=self_similarity_version)#该层的block的第一个卷积核个数64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, self_similarity_radius=self_similarity_radius, self_similarity_version=self_similarity_version)#该层的block的第一个卷积核个数128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, self_similarity_radius=self_similarity_radius, self_similarity_version=self_similarity_version)#该层的block的第一个卷积核个数256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, self_similarity_radius=self_similarity_radius, self_similarity_version=self_similarity_version)#该层的block的第一个卷积核个数512

        reset_weights(self)#手动初始化模型的参数，当然系统也会自动初始化

        self.fc = None
        self.fc_out = fc_out
        if self.fc_out > 0:#有可能不需要全连接层
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, fc_out)
            self.fc_name = 'fc'

    def _make_layer(self, block, planes, blocks, stride=1, self_similarity_radius=None, self_similarity_version=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:#每一层最多只进行一次下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes=planes, stride=stride, downsample=downsample))#要传入stride和downsample
        self.inplanes = planes * block.expansion#输入通道数只在第一个block后改变
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if self_similarity_radius:#相关性没看到是怎么实现的
            if self_similarity_version == 1:
                from . self_sim import SelfSimilarity1
                layers.append(SelfSimilarity1(self_similarity_radius, self.inplanes))
            else:
                from . self_sim import SelfSimilarity2
                layers.append(SelfSimilarity2(self_similarity_radius, self.inplanes))
        return nn.Sequential(*layers)

    def forward(self, x, out_layer=0):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if out_layer==-1:
            return x, self.layer4(x)
        x = self.layer4(x)

        if self.fc_out > 0:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

    def load_pretrained_weights(self, pretrain_code):
        if pretrain_code == 'imagenet':
            model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            }
        else:
            raise NameError("unknown pretraining code '%s'" % pretrain_code)

        print("Loading ImageNet pretrained weights for %s" % pretrain_code)
        assert self.model_name in model_urls, "Unknown model '%s'" % self.model_name

        model_dir='dirtorch/data/models/classification/'
        import os, stat # give group permission
        try: os.makedirs(model_dir)
        except OSError: pass

        import torch.utils.model_zoo as model_zoo
        state_dict = model_zoo.load_url(model_urls[self.model_name], model_dir=model_dir)

        from . import load_pretrained_weights
        load_pretrained_weights(self, state_dict)





def resnet18(out_dim=2048):
    """Constructs a ResNet-18 model.
    """
    net = ResNet(BasicBlock, [2, 2, 2, 2], out_dim, 'resnet18')
    return net

def resnet50(out_dim=2048):
    """Constructs a ResNet-50 model.
    """
    net = ResNet(Bottleneck, [3, 4, 6, 3], out_dim, 'resnet50')
    return net

def resnet101(out_dim=2048):
    """Constructs a ResNet-101 model.
    """
    net = ResNet(Bottleneck, [3, 4, 23, 3], out_dim, 'resnet101')
    return net

def resnet152(out_dim=2048):
    """Constructs a ResNet-152 model.
    """
    net = ResNet(Bottleneck, [3, 8, 36, 3], out_dim, 'resnet152')
    return net
