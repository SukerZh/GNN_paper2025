import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random  # 确保random被导入
from data_utils import load_fixed_splits, eval_acc
from utils import load_dataset, edgeindex_construct
from models import GFK

def set_seed(seed: int):
    """设置所有随机种子保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate(model, feature, label, index, eval_func):
    """模型评估函数"""
    model.eval()
    with torch.no_grad():
        out = model(feature[index])
        return eval_func(label[index], out)

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Cora Evaluation with tau=1')
    parser.add_argument('--seed', type=int, default=51290)
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设备配置
    device = torch.device(f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu')

    # 数据加载与预处理
    data = np.load('data/cora/cora.npz')
    edge_index, features, labels = data['edge_index'], data['feats'], data['labels']
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).squeeze().to(device)

    # 修复点：使用正确的参数名 self_loop ##############
    LP, _, _ = edgeindex_construct(edge_index, labels.shape[0], self_loop=True)
    edge_index = LP  # 使用返回的稀疏矩阵

    # 加载固定数据分割
    split_idx_lst = load_fixed_splits('cora')

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    results = []

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        # 特征传播
        propagated_features, feat_dim = load_dataset(
            edge_index, 
            features, 
            args.K, 
            args.tau,
            homoratio=0.8,  # 示例值，需实际计算
            plain=False
        )

        # 模型初始化
        model = GFK(
            level=args.K,
            nfeat=feat_dim,
            nlayers=args.nlayers,
            nhidden=args.hid,
            nclass=labels.max().item()+1,
            dropoutC=0.5,
            dropoutM=0.5
        ).to(device)

        optimizer = optim.AdamW([
            {'params': model.mlp.parameters(), 'weight_decay': 5e-4},
            {'params': model.comb.parameters(), 'weight_decay': 5e-4}
        ], lr=args.lr)

        # 训练循环
        best_val_acc = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(propagated_features[train_idx])
            loss = criterion(out, labels[train_idx])
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                val_acc = evaluate(model, propagated_features, labels, split_idx['valid'], eval_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'temp_model.pt')

        # 测试评估
        model.load_state_dict(torch.load('temp_model.pt'))
        test_acc = evaluate(model, propagated_features, labels, split_idx['test'], eval_acc)
        results.append(test_acc)
        os.remove('temp_model.pt')

    # 结果输出
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f"Cora (tau=1) | Test Acc: {mean_acc:.2f}% ± {std_acc:.2f}")

if __name__ == '__main__':
    main()