import dgl
import torch
from utils.utils import mp2vec_feat
import pandas as pd
import numpy as np
from datetime import datetime

dgl.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def construct_htg_covid(glist, idx, time_window):
    sub_glist = glist[idx-time_window:idx]

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)

    G_feat = dgl.heterograph(hetero_dict)
    
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']

    G_label = glist[idx]

    return G_feat, G_label


def load_COVID_data(glist, time_window):
    train_feats, train_labels = [], []
    val_feats, val_labels     = [], []
    test_feats, test_labels   = [], []

    for i in range(len(glist)):
        if i >= time_window: # 7
            G_feat, G_label = construct_htg_covid(glist, i, time_window)
            if i >= len(glist)-30 and i <= len(glist)-1:
                test_feats.append(G_feat)
                test_labels.append(G_label)
            elif i >= len(glist)-60 and i <= len(glist)-30:
                val_feats.append(G_feat)
                val_labels.append(G_label)
            else:
                train_feats.append(G_feat)
                train_labels.append(G_label)
    
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels


def construct_htg_mag(glist, idx, time_window):
    sub_glist = glist[idx-time_window:idx] #0:3

    ID_dict = {}

    for ntype in glist[0].ntypes:
        ID_set = set()
        for g_s in sub_glist:
            tmp_set = set(g_s.ndata['_ID'][ntype].tolist())
            ID_set.update(tmp_set)
        ID_dict[ntype] = {ID: idx for idx, ID in enumerate(sorted(list(ID_set)))}

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            ID_src = g_s.ndata['_ID'][srctype]
            ID_dst = g_s.ndata['_ID'][dsttype]
            new_src = ID_src[src]
            new_dst = ID_dst[dst]

            new_new_src = [ID_dict[srctype][e.item()] for e in new_src]
            new_new_dst = [ID_dict[dsttype][e.item()] for e in new_dst]

            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (new_new_src, new_new_dst)
            hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (new_new_dst, new_new_src)

    G_feat = dgl.heterograph(hetero_dict)
    
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            G_feat.nodes[ntype].data[f't{t}'] = torch.zeros(G_feat.num_nodes(ntype), g_s.nodes[ntype].data['feat'].shape[1])
            node_id = g_s.ndata['_ID'][ntype]
            node_feat = g_s.ndata['feat'][ntype]
            for (id, feat) in zip(node_id, node_feat):
                G_feat.nodes[ntype].data[f't{t}'][ID_dict[ntype][id.item()]] = feat

    return G_feat


def generate_APA(graph, device):
    AP = graph.adj(etype=('author', 'writes', 'paper')).to_dense()
    PA = AP.t()
    APA = torch.mm(AP.to(device), PA.to(device)).detach().cpu()
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    
    return APA


def construct_htg_label_mag(glist, idx, device):

    APA_cur = generate_APA(glist[idx], device)
    APA_pre = generate_APA(glist[idx-1], device)
    
    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()
    
    APA_sub = APA_cur - APA_pre # new co-author relation
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5
    
    # get indices of author pairs who collaborate
    indices_true = (APA_sub == 1).nonzero(as_tuple=True)
    indices_false = (APA_add == 0).nonzero(as_tuple=True)
    
    pos_src = indices_true[0]
    pos_dst = indices_true[1]
    
    size = int(pos_src.shape[0] * 0.1)
    
    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    pos_src = pos_src[pos_idx]
    pos_dst = pos_dst[pos_idx] 
    
    neg_src = indices_false[0]
    neg_dst = indices_false[1]

    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    neg_src = neg_src[neg_idx]
    neg_dst = neg_dst[neg_idx]
    
    return dgl.graph((pos_src, pos_dst), num_nodes=APA_cur.shape[0]), dgl.graph((neg_src, neg_dst), num_nodes=APA_cur.shape[0])


def load_MAG_data(glist, time_window, device):
    
    print('loading mp2vec')
    glist = [mp2vec_feat(f'mp2vec/g{i}.vector', g) for (i, g) in enumerate(glist)]

    train_feats, train_labels = [], []
    val_feats, val_labels     = [], []
    test_feats, test_labels   = [], []

    print(f'generating train, val, test sets ')
    for i in range(len(glist)):
        if i >= time_window:
            G_feat = construct_htg_mag(glist, i, time_window)
            pos_label, neg_label = construct_htg_label_mag(glist, i, device)
            if i == len(glist)-1:
                test_feats.append(G_feat)
                test_labels.append((pos_label, neg_label))
            elif i == len(glist)-2:
                val_feats.append(G_feat)
                val_labels.append((pos_label, neg_label))
            else: 
                train_feats.append(G_feat)
                train_labels.append((pos_label, neg_label))
                
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels

def load_ML_data(time_window, device):
    mm = pd.read_csv('/home/jiazhengli/xdgnn/HTGNN/data/Movielens/movie_movie(knn).dat', sep = "\t", header=None,names=['m1', 'm2','score'])
    uu = pd.read_csv('/home/jiazhengli/xdgnn/HTGNN/data/Movielens/user_user(knn).dat', sep = "\t", header=None,names=['u1', 'u2','score'])
    um = pd.read_csv('/home/jiazhengli/xdgnn/HTGNN/data/Movielens/user_movie.dat', sep = "\t", header=None,names=['user', 'movie','rating','time'])

    mmi = np.vstack((mm.m1.values,mm.m2.values))
    uui = np.vstack((uu.u1.values,uu.u2.values))
    mmi = mmi - 1
    uui = uui - 1

    um = um[um.rating>=3]

    um['year'] = um['time'].map(lambda x : datetime.fromtimestamp(x).year) 
    um['month'] = um['time'].map(lambda x : datetime.fromtimestamp(x).month)
    um['day'] = um['time'].map(lambda x : datetime.fromtimestamp(x).day)
    um['t'] = (um['year'] - 1997) * 365 + um['month'] * 30 + um['day']
    um['t'] = (um['t'] // 10)
    um.t = um.t-28
    times = np.unique(um.t)
    um.user = um.user - 1
    um.movie = um.movie - 1

    graph_data = {}
    for t in np.unique(um.t):
        um_t = um[um.t == t]
        u_t = um_t.user.values
        m_t = um_t.movie.values

        graph_data[('user',f'u-m_t{t}','movie')] = (torch.tensor(u_t), torch.tensor(m_t))
        graph_data[('movie',f'm-u_t{t}','user')] = (torch.tensor(m_t), torch.tensor(u_t))
        graph_data[('user',f'u-u_t{t}','user')] = (torch.tensor(uui[0]), torch.tensor(uui[1]))
        graph_data[('movie',f'm-m_t{t}','movie')] = (torch.tensor(mmi[0]), torch.tensor(mmi[1]))
    graph = dgl.heterograph(graph_data).to(device)

    subgraphs = []
    labels = []
    window = 8

    user_feat = torch.randn((943,64)).to(device)
    movie_feat = torch.randn((1682,64)).to(device)

    for i in range(len(times)-window):
        ts = times[i:i+window]
        sg_list = [f'u-m_t{t}' for t in ts] + [f'u-u_t{t}' for t in ts] + [f'm-m_t{t}' for t in ts] + [f'm-u_t{t}' for t in ts]
        sg = graph.edge_type_subgraph(sg_list)
        
        graph_data = {}
        for srctype, etype, dsttype in sg.canonical_etypes:
            src, dst = sg.edges(etype=(srctype, etype, dsttype))
            a, b = etype.split('_')[0], etype.split('_')[1]
            # print(srctype, etype, dsttype)
            # print(a)
            # print(b)
            graph_data[(srctype,a +'_'+ b[0] + str(int(b[1:])-ts[0]),dsttype)] = (src, dst)
        sg_new = dgl.heterograph(graph_data).to(device)

        for t in range(window):
            sg_new.ndata[f't{t}'] = {'user': user_feat[sg_new.nodes('user')] ,'movie': movie_feat[sg_new.nodes('movie')]}

        subgraphs.append(sg_new)
        label_t = np.max(ts) + 1
        # label_g = graph.edge_type_subgraph([f'u-m_t{label_t}', f'm-u_t{label_t}',f'u-u_t{label_t}',f'm-m_t{label_t}'])
        pos = graph.edges(etype=('user', f'u-m_t{label_t}', 'movie'))
        neg1 = torch.from_numpy(np.random.randint(0, graph.num_nodes('user')-1, [len(pos[0])]))
        neg2 = torch.from_numpy(np.random.randint(0, graph.num_nodes('movie')-1, [len(pos[0])]))
        neg = (neg1,neg2)
        label = (pos,neg)
        labels.append(label)

    train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = \
    subgraphs[:10], labels[:10], subgraphs[10:13], labels[10:13], subgraphs[13:], labels[13:]

    # print(train_feats[0].ndata)

    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels