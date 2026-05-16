
from matplotlib import pyplot as plt
from sklearn import manifold
import os
import pickle
import warnings
import time
import random

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from args import parameter_parser
from utils import get_evaluation_results
from Dataloader import load_data
from model import DeepMvNMF, Decomposition, CoGFormer
import pymetis
import scipy.sparse as sp


def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)

# 创建缓存目录（如果不存在）
cache_dir = os.path.join('.', 'cache')
os.makedirs(cache_dir, exist_ok=True)

def train(args, device):
    # 加载输入数据
    feature_list, adj_list, labels, idx_labeled, idx_unlabeled, adj_hats_list = load_data(args, device)

    fused_adj = torch.zeros_like(adj_list[0])
    for adj in adj_list:
        fused_adj += (adj != 0).float()
    fused_adj = (fused_adj > 0).float()

    global_cache_file = os.path.join(
        cache_dir,
        f'global_distance_{args.num_centroids}_{args.dataset}_knn{args.knns}.pkl'
    )

    if os.path.exists(global_cache_file):
        with open(global_cache_file, 'rb') as f:
            global_distance, global_community = pickle.load(f)
            global_distance = torch.tensor(global_distance, dtype=torch.int8).to(device)
            global_community = global_community.to(device)
        print('Loaded cached global distance matrix')
    else:
        fused_adj_np = fused_adj.cpu().numpy()
        edge_index = np.where(fused_adj_np != 0)
        edge_list = list(zip(edge_index[0], edge_index[1]))

        nx_G = nx.Graph()
        nx_G.add_nodes_from(range(fused_adj.shape[0]))
        nx_G.add_edges_from(edge_list)

        # Metis
        adjacency = []
        for node in range(fused_adj.shape[0]):
            neighbors = list(nx_G.neighbors(node))
            adjacency.append(neighbors)

        nparts = args.num_centroids
        cut, parts = pymetis.part_graph(nparts, adjacency=adjacency)
        global_community = torch.tensor(parts, dtype=torch.long)

        super_G = nx.Graph()
        super_G.add_nodes_from(range(nparts))

        for u, v in nx_G.edges():
            comm_u = global_community[u].item()
            comm_v = global_community[v].item()
            if comm_u != comm_v:
                super_G.add_edge(comm_u, comm_v)

        super_distances = dict(nx.all_pairs_shortest_path_length(super_G))
        global_distance = np.full((fused_adj.shape[0], nparts), 30)
        for node in range(fused_adj.shape[0]):
            source_comm = global_community[node].item()
            for target_comm in super_G.nodes():
                if target_comm in super_distances[source_comm]:
                    global_distance[node, target_comm] = super_distances[source_comm][target_comm]

        with open(global_cache_file, 'wb') as f:
            pickle.dump((global_distance, global_community.cpu()), f, pickle.HIGHEST_PROTOCOL)
        print('Processed and cached global distance matrix')
    distance_matrix = torch.tensor(global_distance).to(device)
    nodes_to_community_tensor = torch.tensor(global_community).to(device)

    mask = torch.zeros_like(adj_list[0]).bool().to(device)
    for adj in adj_list:
        mask = mask | adj.bool()
    mask = torch.where(mask, 1, 0)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)

    input_dims = []
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])
    if args.dataset == 'Reuters':
        Defeature = Decomposition(input_dims, 256).to(device)
        x_de = Defeature(feature_list)
        input_dims = []
        for i in range(num_view): # multiview data { data includes features and ... }
            input_dims.append(x_de[i].shape[1])
            feature_list[i] = x_de[i].detach()
        torch.cuda.empty_cache()

    en_hidden_dims = [N, 128]  # 节点数，128
    DMF_model = DeepMvNMF(input_dims, en_hidden_dims, num_view, device).to(device)
    optimizer_DMF = torch.optim.Adam(DMF_model.parameters(), lr=args.share_lr, weight_decay=args.share_weight_decay)
    identity = torch.eye(feature_list[0].shape[0]).to(device)  # 节点数*节点数

    with tqdm(total=1000, desc="Pretraining") as pbar:
        for epoch in range(1000):
            shared_z, x_hat_list = DMF_model(identity)
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            optimizer_DMF.zero_grad()
            loss_DMF.backward()
            optimizer_DMF.step()
            pbar.set_postfix({'Loss': '{:.6f}'.format(loss_DMF.item())})
            pbar.update(1)

    shared_z = shared_z.detach()

    edge_index_A = sp.coo_matrix(mask.cpu())
    indices = np.vstack((edge_index_A.row, edge_index_A.col))

    #MODEL
    model = CoGFormer(
        feature_list=feature_list,
        in_channels=shared_z.shape[1],
        hidden_channels=args.hdim,
        out_channels=num_classes,
        device=device,
        adj_list=adj_list,
        dropout=0.5, num_centroids=args.num_centroids
    ).to(device)
    optimizer_CoGFormer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    Best_Acc=0
    Best_F1=0
    Loss_list = []
    ACC_list = []
    F1_list = []

    begin_time = time.time()

    with tqdm(total=args.num_epoch, desc="training", position=0) as pbar:
        for epoch in range(args.num_epoch):
            shared_z, x_hat_list = DMF_model(identity)
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            loss_share = loss_DMF
            optimizer_DMF.zero_grad()
            loss_share.backward()
            optimizer_DMF.step()

            shared_z = shared_z.detach()

            model.train()
            optimizer_CoGFormer.zero_grad()
            output, _ = model(shared_z, distance_matrix, nodes_to_community_tensor, epoch)
            loss_ce = cross_entropy_loss(output[idx_labeled], labels[idx_labeled])


            loss_ce.backward()
            optimizer_CoGFormer.step()

            # 获取测试结果，取最好的
            with torch.no_grad():
                model.eval()
                result, z = model(shared_z, distance_matrix, nodes_to_community_tensor, 0)
                pred_labels = torch.argmax(result, 1).cpu().detach().numpy()
                ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled],
                                                       pred_labels[idx_unlabeled])

                if ACC > Best_Acc:
                    Best_Acc = ACC
                    Best_P = P
                    Best_R = R
                    Best_F1 = F1

                pbar.set_postfix({'Loss_ce': '{:.6f}'.format(loss_ce.item()),
                                  'ACC': '{:.2f}'.format(ACC * 100), 'Best acc': '{:.4f}'.format(Best_Acc * 100),
                                  'Best F1': '{:.4f}'.format(Best_F1 * 100)})

                pbar.update(1)
                Loss_list.append(float(loss_ce.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)
    cost_time = time.time() - begin_time

    # 取最后一次训练结果
    model.eval()
    result, output_ = model(shared_z, distance_matrix, nodes_to_community_tensor, 0)
    #draw_plt(output_, labels)
    print("Evaluating the model")
    pred_labels = torch.argmax(result, 1).cpu().detach().numpy()
    ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("------------------------")
    print("ratio = ",args.ratio)
    print("ACC:   {:.2f}".format(ACC * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("------------------------")

    return ACC, P, R, F1, cost_time, Loss_list,ACC_list, F1_list, Best_Acc


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()
    save_direction='./adj_matrix/' + args.dataset + '/'
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    args.device = device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.dataset = 'ALOI'
    args.num_centroids = 64
    all_ACC = []
    all_P = []
    all_R = []
    all_F1 = []
    all_TIME = []
    for i in range(args.n_repeated):
        torch.cuda.empty_cache()
        ACC, P, R, F1, Time, Loss_list, ACC_list, F1_list, Best_Acc = train(args, device)
        all_ACC.append(ACC)
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)
        all_TIME.append(Time)

        print("-----------------------")
        print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        print("P  : {:.2f} ({:.2f})".format(np.mean(all_P) * 100, np.std(all_P) * 100))
        print("R  : {:.2f} ({:.2f})".format(np.mean(all_R) * 100, np.std(all_R) * 100))
        print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
        print("-----------------------")


