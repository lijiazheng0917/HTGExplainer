import torch
from torch import nn
import dgl
import torch.nn.functional as F
from utils.utils import TimeEncode

def create_explainer_input_ml(sg, embed, node1, node2, het_emb, etype_dic, tem_emb, device):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        allemb = torch.tensor([]).to(device)
        for srctype, etype, dsttype in sg.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            new_src = sg.ndata[dgl.NID][srctype][src.long()]
            new_dst = sg.ndata[dgl.NID][dsttype][dst.long()]
            srcemb = embed[srctype][new_src.long()] # edge num, embsize
            dstemb = embed[dsttype][new_dst.long()]
            nemb1 = embed['user'][node1].repeat(len(src), 1)
            nemb2 = embed['movie'][node2].repeat(len(src), 1)

            tememb = tem_emb[int(etype[-1])].repeat(len(src), 1)
            hetemb = het_emb[etype_dic[etype[:-3]]].repeat(len(src), 1)

            srcemb = F.normalize(srcemb,dim=1)
            dstemb = F.normalize(dstemb,dim=1)
            tememb = F.normalize(tememb,dim=1)
            hetemb = F.normalize(hetemb.float(),dim=1)
            nemb1 = F.normalize(nemb1,dim=1)
            nemb2 = F.normalize(nemb2,dim=1)

            eemb = torch.cat([srcemb,dstemb,tememb,hetemb,nemb1,nemb2],dim=1)
            allemb = torch.cat([allemb,eemb],dim=0)

        return allemb

def create_explainer_input_mag(sg, embed, node1, node2, het_emb, etype_dic, tem_emb, device):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        allemb = torch.tensor([]).to(device)
        for srctype, etype, dsttype in sg.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            new_src = sg.ndata[dgl.NID][srctype][src.long()]
            new_dst = sg.ndata[dgl.NID][dsttype][dst.long()]
            srcemb = embed[srctype][new_src.long()] # edge num, embsize
            dstemb = embed[dsttype][new_dst.long()]
            nemb1 = embed['author'][node1].repeat(len(src), 1)
            nemb2 = embed['author'][node2].repeat(len(src), 1)
            tememb = tem_emb[int(etype[-1])].repeat(len(src), 1)
            hetemb = het_emb[etype_dic[etype[:-3]]].repeat(len(src), 1)

            srcemb = F.normalize(srcemb,dim=1)
            dstemb = F.normalize(dstemb,dim=1)
            tememb = F.normalize(tememb,dim=1)
            hetemb = F.normalize(hetemb.float(),dim=1)
            nemb1 = F.normalize(nemb1,dim=1)
            nemb2 = F.normalize(nemb2,dim=1)

            eemb = torch.cat([srcemb,dstemb,tememb,hetemb,nemb1,nemb2],dim=1)
            allemb = torch.cat([allemb,eemb],dim=0)

        return allemb

def create_1d_absolute_sin_cos_embedding(pos_len, dim):
        '''
        Create 1D absolute positional encoding.
        :param pos_len: length of the position
        :param dim: dimension of the embedding
        :return: positional embedding
        '''
        assert dim % 2 == 0, "wrong dimension!"
        position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
        i_matrix = torch.arange(dim//2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos matrix
        pos_vec = torch.arange(pos_len).to(torch.long)
        out = pos_vec[:, None] @ i_matrix[None, :]
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        position_emb[:, 0::2] = emb_sin
        position_emb[:, 1::2] = emb_cos
        return position_emb

class HTGNNExplainer(nn.Module):
    def __init__(self, te, he, num_types, node_emb,tw):
        super(HTGNNExplainer, self,).__init__()

        self.he = he
        # create heterogeneous embeddings
        if he == 'learnable':
            self.het_emb = nn.Parameter(torch.zeros(num_types, node_emb))
        elif he == 'onehot':
            self.het_emb = torch.eye(num_types, requires_grad=False)
        else:
            raise ValueError('Unknown heterogeneous embedding type')

        # create temporal embeddings
        if te == 'cos':
            TE = TimeEncode(node_emb)
            self.tem_emb = TE(torch.arange(tw).float())
        elif te == 'pos':
            self.tem_emb = create_1d_absolute_sin_cos_embedding(pos_len=tw, dim=node_emb)
        else:
            raise ValueError('Unknown temporal encoding type')

        self.tem_emb = self.tem_emb.to('cuda')
        # freeze the embeddings
        self.tem_emb.requires_grad = False

        # MLP explainer model
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16,1)
        )
        self.t_linear = nn.Linear(self.tem_emb.shape[1],self.tem_emb.shape[1])
        self.h_linear = nn.Linear(self.het_emb.shape[1],self.het_emb.shape[1])
    
    def forward(self,sg, embed, src, dst, etype_dic,device, dataset):
        # transform the embeddings
        if self.he == 'learnable':
            het_emb = self.h_linear(self.het_emb)
        else:
            het_emb = self.het_emb
            het_emb = het_emb.to('cuda')
        tem_emb = self.t_linear(self.tem_emb)

        # generate the input for the explainer model
        if dataset == 'ml':
            input_expl = create_explainer_input_ml(sg,embed,src,dst,het_emb, etype_dic, tem_emb,device).unsqueeze(0)
        if dataset == 'mag':
            input_expl = create_explainer_input_mag(sg,embed,src,dst,het_emb, etype_dic, tem_emb,device).unsqueeze(0)

        return self.explainer_model(input_expl)


