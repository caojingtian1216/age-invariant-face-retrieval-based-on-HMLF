class Config:
    # --- 核心配置 ---
    
    # 输入图片尺寸
    # FaceNet 推荐: 160 或 224
    # InsightFace 必须: 112
    INPUT_SIZE = 112 

    # 归一化参数
    # FaceNet / InsightFace 通用: [0.5, 0.5, 0.5]
    # ImageNet 预训练通用: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

# 实例化一个全局对象
cfg = Config()