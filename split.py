import os
import numpy as np

# 加载 npz 数据文件，假设节点特征存储在 "node_feat" 字段中
npz_path = "data/cora/cora.npz"
data_npz = np.load(npz_path)
node_feat = data_npz["feats"]

# 获取节点总数
N = node_feat.shape[0]
print("节点总数:", N)

# 按照示例比例划分：train: 87/183, valid: 59/183, test: 37/183
train_ratio = 0.6
valid_ratio = 0.2
train_count = int(round(train_ratio * N))
valid_count = int(round(valid_ratio * N))
test_count = N - train_count - valid_count

print("划分数量 -> train: {}, valid: {}, test: {}".format(train_count, valid_count, test_count))

splits = []
indices = np.arange(N)

# 生成10组随机划分
for i in range(10):
    idx = indices.copy()
    np.random.shuffle(idx)
    train_idx = sorted(idx[:train_count].tolist())
    valid_idx = sorted(idx[train_count:train_count+valid_count].tolist())
    test_idx  = sorted(idx[train_count+valid_count:].tolist())
    split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    splits.append(split)
    print(f"Split {i+1}:")
    print(split)
    print()

# 确保保存路径存在
output_dir = "data/splits"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cora-splits.npy")

# 保存划分结果
np.save(output_path, splits)
print("划分结果已保存到:", output_path)
