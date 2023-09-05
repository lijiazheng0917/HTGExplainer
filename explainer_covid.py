import dgl
from dgl.data.utils import load_graphs
import torch
import torch.nn as nn
import numpy as np
import argparse
# import tensorboard

from model.model_gcn import HTGNN, NodePredictor
from utils.data import load_COVID_data
from model.myPGExplainer_covid import PGExplainer

# parameters
parser = argparse.ArgumentParser(description='explaining HTGNN')
parser.add_argument('--lr', default=5e-3,type=int, help='learning rate')
parser.add_argument('--epoch', default=200,type=int, help='epochs')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--warmup', default=50,type=int, help='warm up epochs')
parser.add_argument('--es', default=20,type=int, help='early stop epochs')
parser.add_argument('--seed', default=42,type=int, help='random seed')
parser.add_argument('--gpu', default=True, help='use gpu or not')
parser.add_argument('--khop', default=1,type=int, help='k-hhop subgraph to explain')
parser.add_argument('--te', default='cos', help='temporal embedding, choose from pos, cos')
parser.add_argument('--he', default='learnable', help='heterogeneous embedding, choose from onehot, learnable')
parser.add_argument('--explain_method', default='our', help='explaining method, choose from our, pg')
parser.add_argument('--test', action='store_true',default=False, help='only test')
args = parser.parse_args()

# seed everything
dgl.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


if torch.cuda.is_available() and args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device
    ('cpu')
print(device)
print(args.khop)
print(args.te)
print(args.he)
print(args.explain_method)
print(args.lr)
print(args.bs)

glist, _ = load_graphs('data/covid_graphs.bin') # len=304
time_window = 15

train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_COVID_data(glist, time_window)

graph_atom = test_feats[0]
model_out_path = 'output/COVID19'
htgnn = HTGNN(graph=graph_atom, n_inp=1, n_hid=8, n_layers=2, n_heads=1, time_window=time_window, norm=False, device=device)
predictor = NodePredictor(n_inp=8, n_classes=1)
model = nn.Sequential(htgnn, predictor).to(device)
model.load_state_dict(torch.load('output/COVID19/checkpoint_HTGNN_0.pt'))

time_window = 15
node_embs = 8
explainer = PGExplainer(model_to_explain = model, G_train = train_feats, G_train_label = train_labels, 
                        G_val = val_feats, G_val_label = val_labels, G_test = test_feats, G_test_label = test_labels, time_win = time_window, node_emb = node_embs,
                        device = device, epochs=args.epoch, lr=args.lr, warmup_epoch = args.warmup, es_epoch = args.es, explain_method = args.explain_method,
                        batch_size = args.bs, khop=args.khop, te=args.te, he=args.he, test_only = args.test)
explainer.explain()
