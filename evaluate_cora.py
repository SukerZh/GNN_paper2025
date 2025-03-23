import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import to_undirected
from utils import load_dataset, edgeindex_construct
from models import GFK
import random

def set_seed(seed: int):
    """设置所有随机种子保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def eval_acc(y_true, y_pred):
    """处理1D标签的准确率计算"""
    y_pred = y_pred.argmax(dim=1)        # 获取预测类别索引
    correct = y_true.eq(y_pred).sum()    # 计算正确预测数
    return correct.item() / y_true.size(0)  # 返回准确率

@torch.no_grad()
def evaluate(model, feature, label, index, eval_func):
    """模型评估函数"""
    model.eval()
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
    parser.add_argument('--dpC', type=float, default=0.5)
    parser.add_argument('--dpM', type=float, default=0.5)
    parser.add_argument('--bias', type=str, default='none')
    parser.add_argument('--sole', action='store_true')
    parser.add_argument('--self_loop', action='store_true', default=True)
    parser.add_argument('--plain', action='store_true')
    args = parser.parse_args()

    # 初始化设置
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.dev}' if torch.cuda.is_available() else 'cpu')

    # 数据加载与处理
    data = np.load('data/cora/cora.npz')
    edge_index, features, labels = data['edge_index'], data['feats'], data['labels']
    
    # 转换为PyTorch张量（保持标签为1D）
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).squeeze().to(device)  # [2708]

    # 构建图结构
    LP, _, _ = edgeindex_construct(edge_index, labels.size(0), args.self_loop)

    # 加载预定义的数据分割（需自行实现）
    def load_fixed_splits(dataset):
        """示例数据分割加载方法，需根据实际文件调整"""
        return [{'train': torch.LongTensor(range(140)),
                 'valid': torch.LongTensor(range(140, 640)),
                 'test': torch.LongTensor(range(640, 1640))} 
                for _ in range(args.runs)]

    split_idx_lst = load_fixed_splits('cora')

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    results = []

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        # 特征传播
        propagated_features, feat_dim = load_dataset(
            LP,
            features,
            args.K,
            args.tau,
            args.plain
        )

        # 模型初始化
        model = GFK(
            level=args.K,
            nfeat=feat_dim,
            nlayers=args.nlayers,
            nhidden=args.hid,
            nclass=labels.max().item() + 1,
            dropoutC=args.dpC,
            dropoutM=args.dpM,
            bias=args.bias,
            sole=args.sole
        ).to(device)

        # 优化器设置
        optimizer = optim.AdamW([
            {'params': model.mlp.parameters(), 'weight_decay': 5e-4},
            {'params': model.comb.parameters(), 'weight_decay': 5e-4}
        ], lr=args.lr)

        # 训练循环
        best_val_acc = 0
        model_path = f'cora_tau1_model_{run}.pth'
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            
            # 前向传播
            out = model(propagated_features[train_idx])
            loss = criterion(out, labels[train_idx])
            
            # 反向传播
            loss.backward()
            optimizer.step()

            # 验证评估
            if epoch % 100 == 0:
                val_acc = evaluate(model, propagated_features, labels, split_idx['valid'], eval_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)

        # 最终测试
        model.load_state_dict(torch.load(model_path))
        test_acc = evaluate(model, propagated_features, labels, split_idx['test'], eval_acc)
        results.append(test_acc)
        os.remove(model_path)

        print(f'Run {run+1}/{args.runs} | '
              f'Best Val: {best_val_acc:.4f} | '
              f'Test Acc: {test_acc:.4f}')

    # 结果报告
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    print(f"\n{'='*40}")
    print(f"Final Results (Cora, τ=1)")
    print(f"Average Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}")
    print(f"Configuration:")
    print(f"- Propagation steps (K): {args.K}")
    print(f"- Hidden dim: {args.hid}")
    print(f"- MLP layers: {args.nlayers}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Runs: {args.runs}")
    print(f"{'='*40}")

if __name__ == '__main__':
    main()