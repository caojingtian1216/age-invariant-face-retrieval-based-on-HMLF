"""
@author: Jun Wang 
@date: 20201019
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module
import torch

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size, out_h, out_w):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        #self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        #self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(4,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h, out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out


def mobilefacenet(pretrained=None, embedding_dim=512, input_size=(112, 112)):
    """
    创建MobileFaceNet模型，兼容项目接口
    
    Args:
        pretrained: 预训练权重路径，如果为None则随机初始化
        embedding_dim: 特征维度，默认512
        input_size: 输入图像尺寸 (H, W)，默认(112, 112)
    
    Returns:
        MobileFaceNet模型
    """
    # 计算经过4次stride=2下采样后的特征图尺寸
    # 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7
    out_h = input_size[0] // 16
    out_w = input_size[1] // 16
    
    model = MobileFaceNet(embedding_size=embedding_dim, out_h=out_h, out_w=out_w)
    
    # 添加必要的属性以兼容项目
    model.iscuda=False
    model.rgb_means = [0.5, 0.5, 0.5]  # MobileFaceNet通常用0.5归一化
    model.rgb_stds = [0.5, 0.5, 0.5]
    model.input_size = (3, input_size[0], input_size[1])
    model.preprocess = dict(
        mean=model.rgb_means,
        std=model.rgb_stds,
        input_size=max(model.input_size)
    )
    model.fc_name = 'linear'  # 用于权重加载
    
    # 加载预训练权重
    if pretrained is not None and pretrained != '':
        model.load_pretrained_weights(pretrained)
    
    return model

    # 为MobileFaceNet类添加load_pretrained_weights方法
def _load_pretrained_weights(self, pretrained_path):
    """
    加载预训练权重
    
    Args:
        pretrained_path: 预训练权重文件路径
    """
    import torch
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 处理不同格式的checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除'module.'、'backbone.'等前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除各种可能的前缀
        k = k.replace('module.', '').replace('backbone.', '')
        new_state_dict[k] = v
    
    # 获取当前模型的state_dict
    model_dict = self.state_dict()
    
    # 过滤掉不匹配的键
    pretrained_dict = {}
    skipped_keys = []
    loaded_keys = []
    
    for k, v in new_state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
                loaded_keys.append(k)
            else:
                print(f"跳过形状不匹配的层: {k}, 预训练: {v.shape}, 模型: {model_dict[k].shape}")
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)
    
    # 更新模型的state_dict
    model_dict.update(pretrained_dict)
    self.load_state_dict(model_dict)
    
    print(f"成功加载 {len(loaded_keys)} 个预训练权重")
    if skipped_keys:
        print(f"跳过 {len(skipped_keys)} 个不匹配的权重")
    
    return self
MobileFaceNet.load_pretrained_weights = _load_pretrained_weights
