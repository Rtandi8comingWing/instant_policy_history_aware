import torch

# 1. 加载权重文件
path = "./model.pt"  # 替换成你的路径
try:
    state_dict = torch.load(path, map_location="cpu")
    print("✅ 成功加载 model.pt")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# 2. 检查这是否是一个包含其他信息的 checkpoint 字典
# 有时候 model.pt 里不只存了权重，还存了 optimizer, epoch 等
if "model_state_dict" in state_dict:
    print("ℹ️ 这是一个 Checkpoint，正在提取 model_state_dict...")
    state_dict = state_dict["model_state_dict"]
elif "state_dict" in state_dict: # 有些人喜欢叫 state_dict
    state_dict = state_dict["state_dict"]

# 3. 打印前 20 个 Key 看看长什么样
print("\n--- 权重文件里的 Key (前20个) ---")
keys = list(state_dict.keys())
for k in keys[:50]:
    shape = state_dict[k].shape
    print(f"{k}: {shape}")

# 4. 专门搜索有没有 'sa1' 或 'encoder' 相关的层
print("\n--- 搜索 Encoder 相关的层 ---")
encoder_keys = [k for k in keys if "scene_encoder" in k]
if encoder_keys:
    print(f"发现 {len(encoder_keys)} 个相关参数，例如：")
    for k in encoder_keys:
        print(f"{k}: {state_dict[k].shape}")
else:
    print("⚠️ 未发现明显的 encoder 关键字，可能命名风格不同。")