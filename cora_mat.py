import scipy.io as sio
from torch_geometric.datasets import Planetoid
import os

# 加载 Cora 数据集，数据将自动保存在当前目录下的 "Cora" 文件夹中
dataset = Planetoid(root='./', name='Cora')
data = dataset[0]

edge_index = data.edge_index.numpy()        # 边索引，形状为 [2, num_edges]
node_feat  = data.x.numpy()                   # 节点特征，形状为 [num_nodes, num_features]
label      = data.y.numpy().squeeze()          # 节点标签，形状为 [num_nodes,]

mat_dict = {
    "edge_index": edge_index,
    "node_feat": node_feat,
    "label": label
}

mat_path = os.path.join("data", "cora.mat")
sio.savemat(mat_path, mat_dict)

print("cora.mat generated successfully!")
print("edge_index shape:", edge_index.shape)
print("node_feat shape:", node_feat.shape)
print("label shape:", label.shape)
