import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear
import numpy as np
import torch.nn.functional as F
import sys

from GCL import GraphConvolution
from utils import compute_renormalized_adj

import warnings
warnings.filterwarnings("ignore")
import math
from einops import rearrange, repeat, reduce

class MLP(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super(MLP, self).__init__()
        # self.fc = nn.Linear(1, 1)
        self.fc = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        # Pass through embedding
        # x = x.unsqueeze(-1)
        # x = torch.relu(self.fc(x))
        # x = x.squeeze(-1)
        # x = self.fc(x)
        x = torch.tanh(self.fc(x))
        return x

    def reset_parameters(self):
        self.fc.reset_parameters()



class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        self.conv1 = GraphConvolution(in_channels, out_channels)
        self.conv2 = GraphConvolution(out_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj, dropout_mask=None):
        x = torch.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, adj)

class CoGFormer(torch.nn.Module):
    def __init__(self, feature_list, in_channels, hidden_channels, out_channels, device,
                 dropout, adj_list, num_centroids=512, heads=4):
        super().__init__()
        self.device = device
        self.adjs = adj_list
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_centroids = num_centroids

        self.attn_fn = F.softmax

        self.lin_key_g = Linear(hidden_channels, hidden_channels)
        self.lin_query_g = Linear(hidden_channels, hidden_channels)
        self.lin_value_g = Linear(hidden_channels, hidden_channels)

        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels * 2, out_channels)  # 使用拼接
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        # 计算视图数
        self.view_num = len(feature_list)
        self.pai = nn.Parameter(torch.ones(self.view_num) / self.view_num, requires_grad=True)

        # 高效阈值生成组件
        self.W = nn.Linear(hidden_channels, 32)
        self.tau = nn.Parameter(torch.tensor([0.5]))

        mlp = MLP(self.num_centroids, self.num_centroids)
        self.fc_dis_list = mlp

        self.conv = GCNBlock(
            hidden_channels,
            hidden_channels
        )


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin_key_g.reset_parameters()
        self.lin_query_g.reset_parameters()
        self.lin_value_g.reset_parameters()
        self.fc_dis.reset_parameters()


    def global_forward(self, x, distance_matrix, nodes_to_community_tensor):
        x_ = x
        # 共享参数初始化
        d, h = self.hidden_channels // self.heads, self.heads
        scale = 1.0 / math.sqrt(d)

        # 基础查询投影
        q = rearrange(self.lin_query_g(x), 'n (h d) -> h n d', h=h)
        distance_matrix = self.fc_dis_list(distance_matrix.float())  # [k, n]
        # print(distance_matrix)
        P = F.one_hot(nodes_to_community_tensor, self.num_centroids).float()
        community_sizes = P.sum(dim=0).view(-1, 1)
        community_sizes = community_sizes.clamp(min=1)
        community_sums = P.T @ x
        comm_avg_sum = community_sums / community_sizes

        k = rearrange(self.lin_key_g(comm_avg_sum), 'k (h d) -> h k d', h=h)
        v = rearrange(self.lin_value_g(comm_avg_sum), 'k (h d) -> h k d', h=h)
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale  # 欧氏内积计算查询q与键k_view的匹配度

        dots += distance_matrix.view(1, distance_matrix.shape[0], distance_matrix.shape[1])  # 距离矩阵偏置

        # 注意力机制
        attn = self.attn_fn(dots, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # 信息聚合
        out = torch.einsum('h i j, h j d -> h i d', attn, v)  # 加权融合
        out = rearrange(out, 'h n d -> n (h d)')  # 多头合并

        return out + x_

    def forward(self, x, distance_matrix, nodes_to_community_tensor, epoch):

        x = self.lin1(x)  # 128-256
        x_ = x
        # 可学习权重进行学习
        exp_sum_pai = 0
        for i in range(self.view_num):
            exp_sum_pai += torch.exp(self.pai[i])
        weight = torch.zeros_like(self.pai)
        for i in range(self.view_num):
            weight[i] = torch.exp(self.pai[i]) / exp_sum_pai
        adj = weight[0] * self.adjs[0]
        for i in range(1, self.view_num):
            adj = adj + weight[i] * self.adjs[i]
        # 高效阈值生成
        H = self.W(x)  # [N, 32]
        thresholds = torch.sigmoid(
            torch.mm(H, H.t()) / 32 + self.tau
        )
        # 边权精炼
        adj = self.DSE(adj, thresholds)
        adj = compute_renormalized_adj(adj, self.device)
        x = self.conv(x, adj)
        x = self.norm(x).relu()
        z_l = F.dropout(x, p=self.dropout, training=self.training)
        z_g = self.global_forward(x_, distance_matrix, nodes_to_community_tensor)
        x = torch.cat([z_l, z_g], dim=-1)

        return self.lin2(x), x

    def DSE(self, adj, thresholds):
        diff_mask = (adj > 0.01) | (thresholds < 0.99)  # 剪枝微小变化
        delta = torch.where(diff_mask, adj - thresholds, torch.zeros_like(adj))

        # （仅计算有显著变化的边）
        shrink_mask = torch.sigmoid(10.0 * delta)
        enhance_mask = 1.0 + F.relu(delta)
        return adj * shrink_mask * enhance_mask







##################################################################################################################

class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=64):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

class Linerlayer(nn.Module):
    def __init__(self, inputdim, outputdim):
        super(Linerlayer, self).__init__()
        self.weight = glorot_init(inputdim, outputdim)
        # self.device = device
    def forward(self, x, sparse=False):
        if sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.mm(x, self.weight)
        return x


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class Decomposition(nn.Module):
    def __init__(self, inputdim_list, outputdim):
        super(Decomposition, self).__init__()
        self.W = nn.ModuleList()
        for i in range(len(inputdim_list)):
            self.W.append(Linerlayer(inputdim_list[i],outputdim))
    def forward(self, feature_list):
        de_feature_list = []
        for i in range(len(feature_list)):
            x = self.W[i](feature_list[i],sparse=True)
            de_feature_list.append(x)
        return de_feature_list

class DeepMvNMF(nn.Module):
    def __init__(self, input_dims, en_hidden_dims, num_views, device):
        super(DeepMvNMF, self).__init__()
        self.encoder = nn.ModuleList()
        self.mv_decoder = nn.ModuleList()
        self.device = device
        # self.decrease = nn.Linear(en_hidden_dims[i], en_hidden_dims[i + 1])
        for i in range(len(en_hidden_dims)-1):
            # self.encoder.append(nn.Linear(en_hidden_dims[i], en_hidden_dims[i+1]))
            self.encoder.append(Linerlayer(en_hidden_dims[i], en_hidden_dims[i+1]))
        for i in range(num_views):
            decoder = nn.ModuleList()
            de_hidden_dims = [input_dims[i]]  # 保存每个视图原本的特征维度
            for k in range(1, len(en_hidden_dims)):
                de_hidden_dims.insert(0, en_hidden_dims[k])
            # print(de_hidden_dims)
            for j in range(len(de_hidden_dims)-1):
                decoder.append(nn.Linear(de_hidden_dims[j], de_hidden_dims[j+1]))
            self.mv_decoder.append(decoder)
        # print(self.encoder)
        # print(self.mv_decoder)

    def forward(self, input):
        z = input
        for layer in self.encoder:
            z = F.relu(layer(z,sparse=True))
        x_hat_list = [] # 根据共享表示z得到解码后的每个视图，方便进行重构损失
        for de in self.mv_decoder:
            x_hat = z
            for layer in de:
                x_hat = F.relu(layer(x_hat))
            x_hat_list.append(x_hat)
        return z, x_hat_list
