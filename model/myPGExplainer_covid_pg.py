import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import dgl
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from utils.data import load_COVID_data
import numpy as np
from random import sample
from utils.pytorchtools import EarlyStopping

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('/home/jiazhengli/xdgnn/HTGNN/output/covid_es_pg')
# writer = SummaryWriter('/home/jiazhengli/xdgnn/HTGNN/output/covid_5')

class PGExplainer():
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, G_train, G_val, G_test, time_win, epochs=200, lr=0.005, temp=(5.0, 2.0), reg_coefs=(0.001, 1),sample_bias=0):
        super().__init__()

        self.model_to_explain = model_to_explain
        self.model_to_explain.eval()
        self.G_train = G_train
        self.G_val = G_val
        self.G_test = G_test

        self.tw = time_win

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.node_emb = 8

        self.expl_embedding = 8+8+8# +4+8

    def create_1d_absolute_sin_cos_embedding(self, pos_len, dim):
        assert dim % 2 == 0, "wrong dimension!"
        position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
        # i矩阵
        i_matrix = torch.arange(dim//2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos matrix
        pos_vec = torch.arange(pos_len).to(torch.long)
        # 矩阵相乘，pos变成列向量，i_matrix变成行向量
        out = pos_vec[:, None] @ i_matrix[None, :]
        # 奇/偶数列
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        # 赋值
        position_emb[:, 0::2] = emb_sin
        position_emb[:, 1::2] = emb_cos
        return position_emb


    def _create_explainer_input(self, sg, embed, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        pos_emb = self.create_1d_absolute_sin_cos_embedding(pos_len=self.tw, dim=8)
        het = {'statestate':torch.tensor([1,0,0,0]),'statecounty':torch.tensor([0,1,0,0]), \
               'countystate':torch.tensor([0,0,1,0]),'countycounty':torch.tensor([0,0,0,1])}

        allemb = torch.tensor([]).to('cuda')
        for srctype, etype, dsttype in sg.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            new_src = sg.ndata[dgl.NID][srctype][src.long()]
            new_dst = sg.ndata[dgl.NID][dsttype][dst.long()]
            srcemb = embed[srctype][new_src.long()] # edge num, embsize
            dstemb = embed[dsttype][new_dst.long()]
            nemb = embed['state'][node_id].repeat(len(src), 1)
            posemb = pos_emb[int(etype[-1])].repeat(len(src), 1).to('cuda')
            hetemb = het[srctype+dsttype].repeat(len(src), 1).to('cuda')

            srcemb = F.normalize(srcemb,dim=1)
            dstemb = F.normalize(dstemb,dim=1)
            posemb = F.normalize(posemb,dim=1)
            hetemb = F.normalize(hetemb.float(),dim=1)
            nemb = F.normalize(nemb,dim=1)

            # with none
            eemb = torch.cat([srcemb,dstemb,nemb],dim=1)
            # with all
            # eemb = torch.cat([srcemb,dstemb,posemb,hetemb,nemb],dim=1)

            allemb = torch.cat([allemb,eemb],dim=0)

        return allemb

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias) # uniform dist
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to('cuda')
            
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = F.l1_loss(masked_pred, original_pred)

        return cce_loss + mask_ent_loss + size_loss, cce_loss, mask_ent_loss, size_loss

    def _mask_graph_new(self, mask, rate):
        
        new_mask = torch.zeros(mask.shape[0]).to('cuda')
        _, idx = torch.sort(mask,descending=True)

        top_idx = idx[:int(rate*len(idx))]
        new_mask[top_idx]=1

        return new_mask

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.train()

    def evaluate(self, explainer_model, G_val, t):
        explainer_model.eval()
        closs = 0
        for i in range(len(G_val)):
            embed = {}
            for ntype in G_val[i].ntypes:
                embed[ntype] = self.model_to_explain[0](G_val[i].to('cuda'), ntype).detach()
            num_of_states = G_val[i].number_of_nodes('state')
            states_list = list(range(num_of_states))
            # states_list = sample(states_list,int(num_of_states/10))
            # states_list = states_list[:int(num_of_states/10)]
            
            for n in states_list:
                n = int(n)
                sg, _ = dgl.khop_in_subgraph(G_val[i], {'state': n}, k=2, store_ids=True)
                    # key
                if(sg.num_nodes(ntype='state') <=1):
                    continue
                    # Similar to the original paper we only consider a subgraph for explaining

                input_expl = self._create_explainer_input(sg, embed, n).unsqueeze(0).to('cuda')

                sampling_weights = self.explainer_model(input_expl)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')
                # shape of mask: num of edge, 0-1 continuous number
                #masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)

                # new_mask = self._mask_graph_new(mask, 0.05)

                h = self.model_to_explain[0](sg.to('cuda'),'state',edge_weight=mask)
                masked_pred = self.model_to_explain[1](h)

                h = self.model_to_explain[0](sg.to('cuda'),'state')
                original_pred = self.model_to_explain[1](h)

                original_pred = original_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]
                masked_pred = masked_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]

                closs_ = F.l1_loss(original_pred, masked_pred)
                # id_loss, closs_, mloss_, sloss_ = self._loss(masked_pred, original_pred, mask, self.reg_coefs)

                closs += closs_.item()
            
            closs += closs_

            return closs
            

    def train(self):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model = self.explainer_model.to('cuda')

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr, weight_decay=1e-4)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        model_out_path = '/home/jiazhengli/xdgnn/HTGNN/output/explainer_covid'

        # Make it larger??
        early_stopping = EarlyStopping(patience=20, verbose=True, path=f'{model_out_path}/checkpoint_covid_es_pg.pt')

        for e in tqdm(range(0, self.epochs)):

            epoch_loss = 0
            epoch_closs = 0
            epoch_mloss = 0
            epoch_sloss = 0

            for i in range(len(self.G_train)):

                embed = {}
                for ntype in self.G_train[i].ntypes:
                    embed[ntype] = self.model_to_explain[0](self.G_train[i].to('cuda'), ntype).detach()
                # self.G_train[i]

                self.explainer_model.train()
                optimizer.zero_grad()
                loss = torch.FloatTensor([0]).detach().to('cuda')
                closs = 0
                mloss = 0
                sloss = 0
                t = temp_schedule(e)

                num_of_states = self.G_train[i].number_of_nodes('state')
                states_list = list(range(num_of_states))
                states_list = sample(states_list,int(num_of_states/30))

                for n in states_list:
                    n = int(n)

                    sg, _ = dgl.khop_in_subgraph(self.G_train[i], {'state': n}, k=2, store_ids=True)
                        # key
                    if(sg.num_nodes(ntype='state') <=1):
                        continue
                        # Similar to the original paper we only consider a subgraph for explaining

                    input_expl = self._create_explainer_input(sg, embed, n).unsqueeze(0).to('cuda')

                    sampling_weights = self.explainer_model(input_expl)

                    mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')
                    # shape of mask: num of edge, 0-1 continuous number
                    #masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)

                    # masked_sg = self._mask_graph(sg, mask) 

                    h = self.model_to_explain[0](sg.to('cuda'),'state',edge_weight=mask)
                    masked_pred = self.model_to_explain[1](h)

                    h = self.model_to_explain[0](sg.to('cuda'),'state')
                    original_pred = self.model_to_explain[1](h)

                    original_pred = original_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]
                    masked_pred = masked_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]

                    id_loss, closs_, mloss_, sloss_ = self._loss(masked_pred, original_pred, mask, self.reg_coefs)
                    loss += id_loss
                    closs += closs_.item()
                    mloss += mloss_.item()
                    sloss += sloss_.item()
                # writer.add_scalar('loss/all_loss', loss, e)
                # writer.add_scalar('loss/closs', closs, e)
                # writer.add_scalar('loss/mloss', mloss, e)
                # writer.add_scalar('loss/sloss', sloss, e)
                # print('train:',e,loss.item())
                # all_loss.append(loss.item())
                # loss.backward()
                # optimizer.step()

                epoch_loss += loss
                epoch_closs += closs
                epoch_mloss += mloss
                epoch_sloss += sloss

            epoch_loss.backward()
            optimizer.step()

            writer.add_scalar('loss/all_loss', epoch_loss, e)
            writer.add_scalar('loss/closs', epoch_closs, e)
            writer.add_scalar('loss/mloss', epoch_mloss, e)
            writer.add_scalar('loss/sloss', epoch_sloss, e)

            # eval
            if e>50:
                eval_loss = self.evaluate(self.explainer_model, self.G_val, t)
                early_stopping(eval_loss, self.explainer_model)
                if early_stopping.early_stop:
                    print("Early stopping", e)
                    break

            # if (e+1) % 10 == 0:

                # test
        mseloss = nn.MSELoss()
        self.explainer_model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_covid_es_pg.pt'))
        self.explainer_model.eval()
        all_results1 = np.array([])
        all_results2 = np.array([])
        for i in range(len(self.G_test)):
            embed = {}
            for ntype in self.G_test[i].ntypes:
                embed[ntype] = self.model_to_explain[0](self.G_test[i].to('cuda'), ntype).detach()
            num_of_states = self.G_test[i].number_of_nodes('state')
            mae = []
            rmse = []
            for n in range(num_of_states):
                n = int(n)
                sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'state': n}, k=2, store_ids=True)
                    # key
                if(sg.num_nodes(ntype='state') <=1):
                    continue
                    # Similar to the original paper we only consider a subgraph for explaining

                input_expl = self._create_explainer_input(sg, embed, n).unsqueeze(0).to('cuda')
                # print(input_expl.shape)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')
                # shape of mask: num of edge, 0-1 continuous number

                h = self.model_to_explain[0](sg.to('cuda'),'state')
                original_pred = self.model_to_explain[1](h)
                original_pred = original_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]

                rate_list = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2]

                for rate in rate_list:
                # masked_sg = self._mask_graph(sg, mask) 
                    new_mask = self._mask_graph_new(mask, rate)

                    h = self.model_to_explain[0](sg.to('cuda'),'state',edge_weight=new_mask)
                    masked_pred = self.model_to_explain[1](h)
                    masked_pred = masked_pred[torch.where(sg.ndata[dgl.NID]['state'] == n)]

                    l1 = F.l1_loss(original_pred, masked_pred)
                    l2 = mseloss(original_pred, masked_pred)
                    mae.append(l1.item()) 
                    rmse.append(l2.item())

            results1 = np.mean(np.array(mae).reshape((-1,len(rate_list))),axis=0)
            results2 = np.mean(np.array(rmse).reshape((-1,len(rate_list))),axis=0)

            all_results1 = np.concatenate((all_results1,results1))
            all_results2 = np.concatenate((all_results2,results2))
        all_results1 = np.mean(all_results1.reshape((-1,len(rate_list))),axis=0)
        all_results2 = np.mean(all_results2.reshape((-1,len(rate_list))),axis=0)
        # for i in range(len(rate_list)):
        #   writer.add_scalar(f'loss/testmae_{rate_list[i]}', results[i], e)
        print(all_results1)
        print(all_results2)
