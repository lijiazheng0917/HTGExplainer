import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorboard

from model.model_gcn import HTGNN, LinkPredictor
from utils.pytorchtools import EarlyStopping
from utils.utils import compute_metric, compute_loss
from utils.data import load_MAG_data

dgl.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
from model.myPGExplainer_mag3 import PGExplainer

glist, label_dict = load_graphs('data/ogbn_graphs.bin')
device = torch.device('cuda')
time_window = 3

train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_MAG_data(glist, time_window, device)

graph_atom = test_feats[0]
model_out_path = 'output/OGBN-MAG'
auc_list, ap_list = [], []
htgnn = HTGNN(graph=graph_atom, n_inp=128, n_hid=32, n_layers=2, n_heads=1, time_window=time_window, norm=True, device=device)
predictor = LinkPredictor(n_inp=32, n_classes=1)
model = nn.Sequential(htgnn, predictor).to(device)
model.load_state_dict(torch.load('/home/jiazhengli/xdgnn/HTGNN/output/OGBN-MAG/checkpoint_HTGNN_0.pt'))


task = 'node'
explainer = PGExplainer(model_to_explain = model, G = train_feats[0], G_label = train_labels[0],  G2 = test_feats[0], G2_label = test_labels[0], task = task, time_win = time_window)
explainer.prepare()
