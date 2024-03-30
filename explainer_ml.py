import dgl
import torch
import torch.nn as nn
import numpy as np
import argparse
# import tensorboard

from model.HTGNN_ml import HTGNN, LinkPredictor_ml
from utils.data import load_ML_data
from model.HTGExplainer_ml import HTGExplainer

def seed_everything(seed):
    dgl.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # load the arguments
    parser = argparse.ArgumentParser(description='explaining HTGNN')
    parser.add_argument('--lr', default=1e-4, help='learning rate')
    parser.add_argument('--epoch', default=200, help='epochs')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--warmup', default=50, help='warm up epochs')
    parser.add_argument('--es', default=20, help='early stop epochs')
    parser.add_argument('--seed', default=42,type=int, help='random seed')
    parser.add_argument('--gpu', default=True, help='use gpu or not')
    parser.add_argument('--khop', default=1,type=int, help='k-hhop subgraph to explain')
    parser.add_argument('--te', default='cos', help='temporal embedding, choose from pos, cos')
    parser.add_argument('--he', default='learnable', help='heterogeneous embedding, choose from onehot, learnable')
    parser.add_argument('--test', action='store_true',default=False, help='only test')
    parser.add_argument('--tw', default=8, type=int, help='time window, follow htgnn setting')
    parser.add_argument('--node_emb', default=32, type=int, help='node embedding size, follow htgnn setting')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')

    # load data
    train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_ML_data(device)

    # load the to-be-explained model -- HTGNN
    graph_atom = test_feats[0]
    htgnn = HTGNN(graph=graph_atom, n_inp=384, n_hid=32, n_layers=2, n_heads=1, time_window=args.tw, norm=True, device=device)
    predictor = LinkPredictor_ml(n_inp=32, n_classes=1)
    model = nn.Sequential(htgnn, predictor).to(device)
    model.load_state_dict(torch.load('HTGNN/checkpoint_HTGNN_ml.pt'))

    # train and test using HTGExplainer
    explainer = HTGExplainer(model_to_explain = model, G_train = train_feats, G_train_label = train_labels, 
                            G_val = val_feats, G_val_label = val_labels, G_test = test_feats, G_test_label = test_labels, time_win = args.tw, node_emb = args.node_embs,
                            device = device, epochs=args.epoch, lr=args.lr, warmup_epoch = args.warmup, es_epoch = args.es,
                            batch_size = args.bs, khop=args.khop, te=args.te, he=args.he, test_only = args.test)
    explainer.explain()

if __name__ == '__main__':
    main()