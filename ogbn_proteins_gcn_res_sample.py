import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from torch.nn.parallel import DataParallel
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from adj_norm import gcn_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from logger import Logger
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
    to_undirected
)

"""
批处理：full-batch
图数据表示方法：SpMM
模型：GCN_res
数据集：ogbn-arxiv
"""

# 加载数据集
dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./proteins/')
# dataset = PygNodePropPredDataset(name='ogbn-products', root='./products/', transform=T.ToSparseTensor())
print(dataset)
data = dataset[0]
print(data)

# 划分数据集
split_idx = dataset.get_idx_split()

# 定义评估器
evaluator = Evaluator(name='ogbn-proteins')
# evaluator = Evaluator(name='ogbn-products')

edge_index = to_undirected(data.edge_index, data.num_nodes)
#edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]
data.edge_index = edge_index

for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[split_idx[split]] = True
    data[f'{split}_mask'] = mask


data.y = data.y.to(torch.float)
data.num_classes = dataset.num_tasks
data.node_species = None
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

data = T.ToSparseTensor()(data)


# 定义网络
# GCN
# class GCNNet(nn.Module):
#     def __init__(self, dataset, hidden=256, num_layers=3):
#         """
#         :param dataset: 数据集
#         :param hidden: 隐藏层维度，默认256
#         :param num_layers: 模型层数，默认为3
#         """
#         super(GCNNet, self).__init__()
#         self.name = 'GCN_full'
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()

#         self.convs.append(GCNConv(dataset.num_node_features, hidden))
#         self.bns.append(nn.BatchNorm1d(hidden))

#         for i in range(self.num_layers - 2):
#             self.convs.append(GCNConv(hidden, hidden))
#             self.bns.append(nn.BatchNorm1d(hidden))

#         self.convs.append(GCNConv(hidden, dataset.num_classes))

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, data):
#         x, adj_t = data.x, data.adj_t
#         sample1_adj, sample2_adj = sampling(adj_t)
        
#         for i in range(self.num_layers - 1):
#             if i == 0 or i == 1:
#                 x = self.convs[i](x, sample1_adj)
#             else:
#                 x = self.convs[i](x, sample2_adj)
#             x = self.bns[i](x)  # 小数据集不norm反而效果更好
#             x = F.relu(x)
#             x = F.dropout(x, p=0.5, training=self.training)

#         x = self.convs[-1](x, sample2_adj)
#         x = F.log_softmax(x, dim=1)

#         return x

class GCNNet(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=6):
        super(GCNNet, self).__init__()
        self.name = 'GCN_full'
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(dataset.num_node_features, hidden)

        for i in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.out_fc = nn.Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, data):
        x, sample1_adj, sample2_adj = data.x, data.sample1_adj, data.sample2_adj
        x = self.input_fc(x)
        x_input = x  # .copy()

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            if i < self.num_layers/2:
                x = self.convs[i](x, sample1_adj)
            else:
                x = self.convs[i](x, sample2_adj)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)

        return x

# GCN_res
class GCN_res(nn.Module):
    def __init__(self,dataset, hidden=256, num_layers=6):
        super(GCN_res, self).__init__()
        self.name = 'GCN_res'
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(dataset.num_features, hidden)

        for i in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.out_fc = nn.Linear(hidden, dataset.num_classes)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, data):
        x, sample1_adj, sample2_adj = data.x, data.sample1_adj, data.sample2_adj
        x = self.input_fc(x)
        x_input = x  # .copy()

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            if i < self.num_layers/2:
                x = self.convs[i](x, sample1_adj)
            else:
                x = self.convs[i](x, sample2_adj)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)

            # if i == 0:
            #     x = x + 0.2 * x_input
            # else:
            #     x = x + 0.2 * x_input + 0.5 * layer_out[i - 1]
            layer_out.append(x)

        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]

        x = sum(layer_out)
        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)

        return x

# 实例化模型
num_layers = 8
# model = GCNNet(dataset=dataset, hidden=128, num_layers=num_layers)
model = GCN_res(data, hidden=128, num_layers=num_layers)
print(model)

# 转换为cpu或cuda格式
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)
data.edge_index = to_undirected(data.edge_index)
total_edge_index = data.edge_index
data_test = T.ToSparseTensor()(data)
train_idx = train_idx.to(device)

data_test = data
data_test.sample1_adj = data_test.adj_t
data_test.sample2_adj = data_test.adj_t

# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)

# def sampling(edge_index,p_sampling):
#     norm_edge, norm_value = gcn_norm(edge_index, add_self_loops=False)
#     num_values_to_keep = int(norm_value.numel() * p_sampling) #sample less 0.5
#     _, top_indices = torch.topk(norm_value, k=num_values_to_keep)
#     edge_index = norm_edge[:,top_indices]
#     return edge_index
def sampling(edge_index,p_sampling):
    # p_random = 0.1
    norm_edge, norm_value = gcn_norm(edge_index, add_self_loops=False)
    num_values_to_keep = int(norm_value.numel() * p_sampling) #sample less 0.5
    num_values_to_delete = norm_value.numel() - num_values_to_keep
    k=num_values_to_keep
    top_values, top_indices = torch.topk(norm_value, k=num_values_to_keep)
    other_indices = torch.nonzero(torch.lt(norm_value, top_values[k-1])).squeeze()
    
    if num_values_to_keep>=num_values_to_delete:
        num_values_to_random = int(num_values_to_delete*p_random)+1
    
        random_indices = torch.randperm(len(top_indices))[:-num_values_to_random]
        top_indices = top_indices[random_indices]
    
        random_indices = torch.randperm(len(other_indices))[:num_values_to_random]
        other_indices = other_indices[random_indices]

        choose_indices = torch.cat((top_indices, other_indices))
    else:
        num_values_to_random = int(num_values_to_keep*p_random)+1
    
        random_indices = torch.randperm(len(top_indices))[:-num_values_to_random]
        top_indices = top_indices[random_indices]
    
        random_indices = torch.randperm(len(other_indices))[:num_values_to_random]
        other_indices = other_indices[random_indices]

        choose_indices = torch.cat((top_indices, other_indices))
        
    edge_index = norm_edge[:,choose_indices]
    return edge_index

# 定义训练函数
def train():
    model.train()

    out = model(data)
    loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# 定义测试函数
@torch.no_grad()
def test():
    model.eval()

    out = model(data_test)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

# 程序入口
if __name__ == '__main__':
    runs = 10
    epochs = 800
    p_random = 0.1
    p1_sampling_values = np.arange(0.1, 1.01, 0.1)
    p2_sampling_values = np.arange(0.1, 1.01, 1)
    for p1_sampling in p1_sampling_values:
        for p2_sampling in p2_sampling_values:
            # p1_sampling = p1_sampling_value[i] #sample rate in shallow
            # p1_sampling = 0.1 #sample rate in deep
            print(p1_sampling, p2_sampling)
            logger = Logger(runs)
            run_test_accs = [[] for _ in range(runs)]
            
            start_time = time.time()
            
            for run in range(runs):
                print(sum(p.numel() for p in model.parameters()))
                model.reset_parameters()
                # model.module.reset_parameters()
                data.edge_index = sampling(total_edge_index, p1_sampling)
                data.sample1_adj = T.ToSparseTensor()(data).adj_t

                data.edge_index = sampling(total_edge_index, p2_sampling)
                data.sample2_adj = T.ToSparseTensor()(data).adj_t
                for epoch in range(epochs):
                    loss = train()
                    # print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss))
                    if epoch%500==0:
                        lr = 1e-5
                    result = test()
                    train_acc, valid_acc, test_acc = result
                    run_test_accs[run].append(test_acc)
                    # print(f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, 'f'Test: {test_acc:.4f}')
                    print(f'Run: {run + 1:02d}, '
                        f'Epoch: {epoch:02d}, '
                        f'Loss: {loss:.4f}, '
                        f'Train: {100 * train_acc:.2f}%, '
                        f'Valid: {100 * valid_acc:.2f}% '
                        f'Test: {100 * test_acc:.2f}%')

                    logger.add_result(run, result)
            logger.print_statistics()
            # mean_acc = np.mean(run_test_accs, axis=0)
            mean_acc = run_test_accs
            
            end_time = time.time()
            execution_time = end_time - start_time
            print("代码执行时间：", execution_time, "秒")

            # np.save(f'p1={p1_sampling} p2={p2_sampling} GCNFull_{num_layers}_run_test_accs.npy', run_test_accs)
            np.save(f'sampling4/{num_layers}.{model.name}.{p_random}: p1={p1_sampling:.1f} p2={p2_sampling:.1f} runs={runs}.npy', mean_acc)
