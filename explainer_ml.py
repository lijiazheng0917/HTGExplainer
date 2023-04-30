import dgl
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import tensorboard

from model.model_gcn_ml import HTGNN, LinkPredictor_ml
from utils.data import load_ML_data

dgl.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
from model.myPGExplainer_ml import PGExplainer

device = torch.device('cuda')
time_window = 8
train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_ML_data(time_window, device)


graph_atom = test_feats[0]
model_out_path = 'output/ML'
htgnn = HTGNN(graph=graph_atom, n_inp=384, n_hid=32, n_layers=2, n_heads=1, time_window=time_window, norm=True, device=device)
predictor = LinkPredictor_ml(n_inp=32, n_classes=1)
model = nn.Sequential(htgnn, predictor).to(device)
model.load_state_dict(torch.load('/home/jiazhengli/xdgnn/HTGNN/output/ML/checkpoint_HTGNN_0_onehot.pt'))


explainer = PGExplainer(model_to_explain = model, G_train = train_feats, G_train_label = train_labels, 
                        G_val = val_feats, G_val_label = val_labels, G_test = test_feats, G_test_label = test_labels, time_win = time_window)
explainer.prepare()
