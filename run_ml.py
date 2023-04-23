import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.model_gcn import HTGNN, LinkPredictor_ml
from utils.pytorchtools import EarlyStopping
from utils.utils import compute_metric, compute_loss
from utils.data import load_ML_data

dgl.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def evaluate(model, val_feats, val_labels):
    val_mae_list, val_rmse_list = [], []
    model.eval()
    with torch.no_grad():
        for (G_feat, (pos_label, neg_label)) in zip(val_feats, val_labels):

            G_feat = G_feat.to(device)

            # pos_label = pos_label.to(device)
            # neg_label = neg_label.to(device)

            h_u = model[0](G_feat, 'user')
            h_m = model[0](G_feat, 'movie')

            pos_u, pos_m = pos_label[0], pos_label[1]
            neg_u, neg_m = neg_label[0], neg_label[1]

            pos_score, neg_score = model[1](h_u, h_m, pos_u, pos_m, neg_u, neg_m)

            loss = compute_loss(pos_score, neg_score, device)
            auc, ap = compute_metric(pos_score, neg_score)
    
    return auc, ap, loss



device = torch.device('cuda')
time_window = 8

train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_ML_data(time_window, device)

# print(train_feats[0].ndata)

graph_atom = test_feats[0]
model_out_path = 'output/ML'
auc_list, ap_list = [], []

for k in range(1):
    htgnn = HTGNN(graph=graph_atom, n_inp=384, n_hid=32, n_layers=2, n_heads=1, time_window=time_window, norm=True, device=device)
    predictor = LinkPredictor_ml(n_inp=32, n_classes=1)
    model = nn.Sequential(htgnn, predictor).to(device)

    print(f'---------------Repeat time: {k+1}---------------------')
    print(f'# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    early_stopping = EarlyStopping(patience=30, verbose=True, path=f'{model_out_path}/checkpoint_HTGNN_{k}_onehot.pt')
    for epoch in range(500):
        model.train()
        for (G_feat, (pos_label, neg_label)) in zip(train_feats, train_labels):

            G_feat = G_feat.to(device)

            # pos_label = pos_label.to(device)
            # neg_label = neg_label.to(device)

            # print(G_feat)

            h_u = model[0](G_feat, 'user')
            h_m = model[0](G_feat, 'movie')
            # h_u = torch.ones((944,32),device='cuda')
            # h_m = torch.ones((1683,32),device='cuda')

            pos_u, pos_m = pos_label[0], pos_label[1]
            neg_u, neg_m = neg_label[0], neg_label[1]

            pos_score, neg_score = model[1](h_u, h_m, pos_u, pos_m, neg_u, neg_m)

            # pos_score = model[1](pos_label, h)
            # neg_score = model[1](neg_label, h)
            
            loss = compute_loss(pos_score, neg_score, device)
            auc, ap = compute_metric(pos_score, neg_score)

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        auc, ap, val_loss = evaluate(model, val_feats, val_labels)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_HTGNN_{k}_onehot.pt'))
    auc, ap, test_loss = evaluate(model, test_feats, test_labels)

    print(f'auc: {auc}, ap: {ap}')
    auc_list.append(auc)
    ap_list.append(ap)

# import statistics

# print(f'AUC: {statistics.mean(auc_list)}, {statistics.stdev(auc_list)}')
# print(f'AP: {statistics.mean(ap_list)}, {statistics.stdev(ap_list)}')