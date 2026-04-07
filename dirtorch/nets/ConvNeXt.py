import timm
import torch.nn as nn
import torch
import os

class TimmModel(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True, embedding_dim=512):
        super().__init__()
        # num_classes=embedding_dim 会让模型最后输出我们想要的维度 (例如 512)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=embedding_dim)
        
        # 定义预处理参数 (timm 模型通常需要 ImageNet 的标准化参数)
        # 注意：大部分现代模型需要 224x224 的输入
        self.rgb_means=[0.485, 0.456, 0.406]
        self.rgb_stds=[0.229, 0.224, 0.225]
        self.input_size=(3, 224, 224)
        self.preprocess = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            input_size=224 
        )
        self.iscuda = False # 兼容你代码里的检查逻辑

    def forward(self, x):
        return self.model(x)

def get_model(model_name='convnext_tiny', pretrained=True, embedding_dim=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TimmModel(model_name=model_name, pretrained=pretrained, embedding_dim=embedding_dim)
    DIR_ROOT=os.environ['DIR_ROOT']
    weight_path = os.path.join(DIR_ROOT, 'dirtorch/data/models/convnext_tiny_1k_224_ema.pth')

    # 3. 加载权重
    print(f"正在加载本地权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)

    # 4. 提取参数字典 (官方权重通常包在 'model' 键里，但也可能是直接的字典)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    keys_to_remove = [k for k in state_dict.keys() if 'head' in k]
    for k in keys_to_remove:
        print(f"忽略预训练权重层: {k}")
        del state_dict[k]

    # 6. 载入权重 (strict=False 允许我们刚才删掉的 head 层不匹配)
    msg = net.load_state_dict(state_dict, strict=False)
    return net