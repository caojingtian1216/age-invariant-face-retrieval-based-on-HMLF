from huggingface_hub import HfApi, login

# 1. 直接在代码里完成登录（把这里换成你的 hf_ 开头的 Write 权限 Token）
login(token="YOUR_HF_TOKEN_HERE")

# 2. 初始化 API
api = HfApi()

# 3. 设置你的仓库 ID 和本地文件夹路径
repo_id = "caojingtian1216/Swin-Transformer"
local_folder_path = "C:/Users/surface/Desktop/deep-image-retrieval-master/dirtorch/data/models/backbone.pth"

print(f"开始上传 {local_folder_path} 到 {repo_id}...")

# 4. 执行上传 (这部分代码和之前一样)
try:
    api.upload_file(
        path_or_fileobj="C:/Users/surface/Desktop/deep-image-retrieval-master/dirtorch/data/models/swin_tiny_patch4_window7_224.pt", # 本地单个文件的具体路径
        path_in_repo="swin_tiny_patch4_window7_224.pt",                        # 上传到 Hugging Face 仓库后，你想让它叫什么名字
        repo_id=repo_id,
        repo_type="model",                                             # 如果你是在向数据集仓库传单个文件，这里改为 "dataset"
    )
    print("\n✅ 所有文件上传完成！")
except Exception as e:
    print(f"\n❌ 上传出错: {e}")