import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorboard

from model.model_gcn import HTGNN, NodePredictor
from utils.pytorchtools import EarlyStopping
from utils.data import load_COVID_data

dgl.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
from model.myPGExplainer_covid import PGExplainer

device = 'cuda:0'
glist, _ = load_graphs('data/covid_graphs.bin') # len=304
time_window = 15

train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_COVID_data(glist, time_window)


graph_atom = test_feats[0]
model_out_path = 'output/COVID19'
k = 4
htgnn = HTGNN(graph=graph_atom, n_inp=1, n_hid=8, n_layers=2, n_heads=1, time_window=time_window, norm=False, device=device)
predictor = NodePredictor(n_inp=8, n_classes=1)
model = nn.Sequential(htgnn, predictor).to(device)
model.load_state_dict(torch.load('/home/jiazhengli/xdgnn/HTGNN/output/COVID19/checkpoint_HTGNN_0.pt'))

time_window = 15
task = 'node'
# explainer = PGExplainer(model_to_explain = model, G = train_feats[0], G2 = test_feats[1], task = task, time_win = time_window)
explainer = PGExplainer(model_to_explain = model, G_train = train_feats, G_val = val_feats, G_test = test_feats, time_win = time_window)
explainer.prepare()
