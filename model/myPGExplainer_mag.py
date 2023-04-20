import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import dgl
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import random
from random import sample
from utils.pytorchtools import EarlyStopping

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('/home/jiazhengli/xdgnn/HTGNN/output/mag_es_pg')


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
    def __init__(self, model_to_explain, G_train, G_train_label, G_val, G_val_label, G_test, G_test_label, time_win, epochs=200, lr=0.001, temp=(5.0, 2.0), reg_coefs=(0.0001, 0.01),sample_bias=0):
        super().__init__()

        self.model_to_explain = model_to_explain
        self.model_to_explain.eval()
        self.G_train = G_train
        self.G_train_label = G_train_label
        self.G_val = G_val
        self.G_val_label = G_val_label
        self.G_test = G_test
        self.G_test_label = G_test_label
        self.tw = time_win
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.node_emb = 8

        self.expl_embedding = 32+32+32+32# +32+7

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


    def _create_explainer_input(self, sg, embed, node1, node2):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        

        pos_emb = self.create_1d_absolute_sin_cos_embedding(pos_len=self.tw, dim=32)
        het = {'authorinstitution':torch.tensor([1,0,0,0,0,0,0]),'authorpaper':torch.tensor([0,1,0,0,0,0,0]), \
               'field_of_studypaper':torch.tensor([0,0,1,0,0,0,0]),'institutionauthor':torch.tensor([0,0,0,1,0,0,0]), \
               'paperpaper':torch.tensor([0,0,0,0,1,0,0]),'paperfield_of_study':torch.tensor([0,0,0,0,0,1,0]), \
               'paperauthor':torch.tensor([0,0,0,0,0,0,1])}
        het1 = {'authorinstitution':torch.tensor(0),'authorpaper':torch.tensor(1), \
               'field_of_studypaper':torch.tensor(2),'institutionauthor':torch.tensor(3), \
               'paperpaper':torch.tensor(4),'paperfield_of_study':torch.tensor(5), \
               'paperauthor':torch.tensor(6)}

        allemb = torch.tensor([]).to('cuda')
        for srctype, etype, dsttype in sg.canonical_etypes:
            src, dst = sg.edges(etype=etype)
            new_src = sg.ndata[dgl.NID][srctype][src.long()]
            new_dst = sg.ndata[dgl.NID][dsttype][dst.long()]
            srcemb = embed[srctype][new_src.long()] # edge num, embsize
            dstemb = embed[dsttype][new_dst.long()]
            nemb1 = embed['author'][node1].repeat(len(src), 1)
            nemb2 = embed['author'][node2].repeat(len(src), 1)
            posemb = pos_emb[int(etype[-1])].repeat(len(src), 1).to('cuda')
            hetemb = het[srctype+dsttype].repeat(len(src), 1).to('cuda')
            # hetemb = het_emb(het1[srctype+dsttype]).repeat(len(src), 1).to('cuda')

            srcemb = F.normalize(srcemb,dim=1)
            dstemb = F.normalize(dstemb,dim=1)
            posemb = F.normalize(posemb,dim=1)
            hetemb = F.normalize(hetemb.float(),dim=1)
            nemb1 = F.normalize(nemb1,dim=1)
            nemb2 = F.normalize(nemb2,dim=1)

            # with none
            eemb = torch.cat([srcemb,dstemb,nemb1,nemb2],dim=1)
            # with all
            # eemb = torch.cat([srcemb,dstemb,posemb,hetemb,nemb1,nemb2],dim=1)

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


    def _loss(self, masked_pred, target, mask, reg_coefs):
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

        cce_loss = F.binary_cross_entropy_with_logits(masked_pred,target)

        return cce_loss + mask_ent_loss + size_loss, cce_loss, mask_ent_loss, size_loss

    # def _mask_graph(self, sg, mask):
    #     l = {}
    #     k = 0
    #     for srctype, etype, dsttype in sg.canonical_etypes:
    #         src, dst = sg.edges(etype=etype)
    #         # len(src)
    #         for j in range(k,k+len(src)):
    #             l[j] = [etype,j-k,src[j-k],dst[j-k]]
    #         k += len(src)

    #     _, idx = torch.sort(mask,descending=True)
    #     top_idx = idx[:int(0.1*len(idx))]

    #     masked_sg = sg

    #     for srctype, etype, dsttype in sg.canonical_etypes:
    #         eids = sg.edges(form='eid',etype=etype)
    #         masked_sg = dgl.remove_edges(masked_sg,eids,etype)

    #     # c = {type:0 for type in sg.etypes}

    #     for i in range(len(top_idx)):
    #         type = l[top_idx[i].item()][0]
    #         src = l[top_idx[i].item()][2]
    #         dst = l[top_idx[i].item()][3]
    #         masked_sg.add_edges(src,dst,etype=type)

    #     return masked_sg
    
    def _mask_graph_new(self, mask, rate):
        
        new_mask = torch.zeros(mask.shape[0]).to('cuda')
        _, idx = torch.sort(mask,descending=True)

        top_idx = idx[:int(rate*len(idx))]
        new_mask[top_idx]=1

        return new_mask

    def evaluate(self, explainer_model, G_val, G_val_label, t):
        explainer_model.eval()
        loss = 0
        for i in range(len(G_val)):
            embed = {}
            for ntype in G_val[i].ntypes:
                embed[ntype] = self.model_to_explain[0](G_val[i].to('cuda'), ntype).detach()

            org_feat2 = self.model_to_explain[0](self.G_val[i].to('cuda'),'author')
            pos_label2, neg_label2 = G_val_label[i][0].to('cuda'), G_val_label[i][1].to('cuda')
            pos_score2 = self.model_to_explain[1](pos_label2, org_feat2)
            neg_score2 = self.model_to_explain[1](neg_label2, org_feat2)
            ori_pred2 = torch.cat((pos_score2.squeeze(1), neg_score2.squeeze(1)))
            target2 = torch.sigmoid(ori_pred2).detach()

            num_edges = G_val_label[i][0].num_edges()
            edge_list = list(range(num_edges))
            # how many edges to test?
            # edge_list = sample(edge_list, 1000)
            edge_list = edge_list[:500]
                
            for n in edge_list:
                n = int(n)

                flag = random.choice([0,1])
                src, dst = G_val_label[i][flag].edges()[0][n], G_val_label[i][flag].edges()[1][n]

                sg, _ = dgl.khop_in_subgraph(G_val[i], {'author': (src,dst)}, k=2, store_ids=True)

                input_expl = self._create_explainer_input(sg, embed, src, dst).unsqueeze(0).to('cuda')

                sampling_weights = self.explainer_model(input_expl)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')

                # new_mask = self._mask_graph_new(mask, 0.005) 

                h_m = self.model_to_explain[0](sg.to('cuda'),'author',edge_weight=mask)

                src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                all_h = torch.cat((src_h, dst_h),dim=1).to('cuda')
                pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                
                if flag == 0:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n])
                else:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n+num_edges])
                # size_loss = torch.sum(mask) * size_reg + torch.sum(mask) * size_reg
                # mask_ent_reg1 = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
                # mask_ent_reg2 = -dst_mask * torch.log(dst_mask) - (1 - dst_mask) * torch.log(1 - dst_mask)
                # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg1)# + entropy_reg * torch.mean(mask_ent_reg2)
                loss_ = cce_loss.item() # + size_loss + mask_ent_loss

                loss += loss_      
                
        return loss

    def prepare(self):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        # self.het_emb = nn.Embedding(7,32)
        self.train()

    def train(self):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model = self.explainer_model.to('cuda')

        # Create optimizer and temperature schedule
        # optimizer = Adam([*self.explainer_model.parameters(),*self.het_emb.parameters()], lr=self.lr, weight_decay=1e-4)
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr, weight_decay=1e-4)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        model_out_path = '/home/jiazhengli/xdgnn/HTGNN/output/explainer_mag'
        # early_stopping
        early_stopping = EarlyStopping(patience=20, verbose=True, path=f'{model_out_path}/checkpoint_mag_es.pt')

        size_reg = self.reg_coefs[0]
        entropy_reg = self.reg_coefs[1]

        rate_list = [0.001,0.003,0.005,0.01,0.03,0.05]

        # Start training loop
        for e in tqdm(range(0, self.epochs)):

            self.explainer_model.train()
            t = temp_schedule(e)
            epoch_loss = 0
            epoch_closs = 0
            epoch_mloss = 0
            epoch_sloss = 0

            for i in range(len(self.G_train)):

                org_feat = self.model_to_explain[0](self.G_train[i].to('cuda'),'author')
                pos_label, neg_label = self.G_train_label[i][0].to('cuda'), self.G_train_label[i][1].to('cuda')
                pos_score = self.model_to_explain[1](pos_label, org_feat)
                neg_score = self.model_to_explain[1](neg_label, org_feat)
                ori_pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
                target = torch.sigmoid(ori_pred).detach()

                embed = {}
                # If we are explaining a graph, we can determine the embeddings before we run
                for ntype in self.G_train[i].ntypes:
                    embed[ntype] = self.model_to_explain[0](self.G_train[i].to('cuda'), ntype).detach()

                
                num = self.G_train_label[i][0].num_edges()
                batchs = num // 32
                start = random.choice(list(range(batchs-5)))
                # batchs = 3
                all_loss = 0
                all_closs = 0
                all_mloss = 0
                all_sloss = 0

                # change here
                for j in range(0, 10):
                    optimizer.zero_grad()
                    # loss = torch.FloatTensor([0]).detach().to('cuda')
                    loss = 0
                    closs = 0
                    mloss = 0
                    sloss = 0

                    lo = 8*j
                    # hi = min(16*(j+1),self.G_train_label[i][0].num_edges())
                    hi = 8*(j+1)
                    # print(lo,hi)
                    for n in range(lo, hi):

                        n = int(n)
                        # print(n)
                        flag = random.choice([0,1])

                        src, dst = self.G_train_label[i][flag].edges()[0][n], self.G_train_label[i][flag].edges()[1][n]

                        sg, _ = dgl.khop_in_subgraph(self.G_train[i], {'author': (src,dst)}, k=2, store_ids=True)

                        input_expl = self._create_explainer_input(sg, embed, src, dst).unsqueeze(0).to('cuda')

                        sampling_weights = self.explainer_model(input_expl)

                        mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')

                        h_m = self.model_to_explain[0](sg.to('cuda'),'author',edge_weight=mask)

                        src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                        dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                        all_h = torch.cat((src_h, dst_h),dim=1).to('cuda')
                        pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                        
                        if flag == 0:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n])
                        else:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n+num])

                        size_loss = torch.sum(mask) * size_reg + torch.sum(mask) * size_reg
                        mask_ent_reg1 = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
                        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg1)# + entropy_reg * torch.mean(mask_ent_reg2)
                        loss_ = cce_loss + size_loss + mask_ent_loss
                        
                        
                        loss += loss_
                        closs += cce_loss.item()
                        sloss += size_loss.item()
                        mloss += mask_ent_loss.item()

                    # print("1:{}".format(torch.cuda.memory_allocated(0)))
                    loss.backward()
                    optimizer.step()

                # print(":{}".format(torch.cuda.memory_allocated(0)))

                    all_loss += loss.item()
                    all_closs += closs
                    all_mloss += mloss
                    all_sloss += sloss
            
                epoch_loss += all_loss
                epoch_closs += all_closs
                epoch_mloss += all_mloss
                epoch_sloss += all_sloss


            writer.add_scalar('loss/all_loss', epoch_loss, e)
            writer.add_scalar('loss/closs', epoch_closs, e)
            writer.add_scalar('loss/mloss', epoch_mloss, e)
            writer.add_scalar('loss/sloss', epoch_sloss, e)

            torch.cuda.empty_cache()
        
            if e>50:
                eval_loss = self.evaluate(self.explainer_model, self.G_val, self.G_val_label, t)
                early_stopping(eval_loss, self.explainer_model)
                if early_stopping.early_stop:
                    print("Early stopping", e)
                    break

            
            # if (e+1) % 5 == 0:


        # testing
        # t = temp_schedule(100)

        self.explainer_model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_mag_es.pt'))
        self.explainer_model.eval()
        # self.het_emb.eval()
        all_pred = []
        target_list = []
        for i in range(len(self.G_test)):
            embed = {}
            for ntype in self.G_test[i].ntypes:
                embed[ntype] = self.model_to_explain[0](self.G_test[i].to('cuda'), ntype).detach()

            org_feat2 = self.model_to_explain[0](self.G_test[i].to('cuda'),'author')
            pos_label2, neg_label2 = self.G_test_label[i][0].to('cuda'), self.G_test_label[i][1].to('cuda')
            pos_score2 = self.model_to_explain[1](pos_label2, org_feat2)
            neg_score2 = self.model_to_explain[1](neg_label2, org_feat2)
            ori_pred2 = torch.cat((pos_score2.squeeze(1), neg_score2.squeeze(1)))
            target2 = torch.sigmoid(ori_pred2).detach()

            num_edges = self.G_test_label[i][0].num_edges()
            edge_list = list(range(num_edges))
            # how many edges to test?
            # edge_list = sample(edge_list, 1000)
            edge_list = edge_list[:1000]

            for n in edge_list:
                n = int(n)
                # print(n)
                target_list.append(torch.round(target2[n]).detach().cpu().numpy())

                src, dst = self.G_test_label[i][0].edges()[0][n], self.G_test_label[i][0].edges()[1][n]

                sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'author': (src,dst)}, k=2, store_ids=True)

                input_expl = self._create_explainer_input(sg, embed, src, dst).unsqueeze(0).to('cuda')

                sampling_weights = self.explainer_model(input_expl)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')

                for rate in rate_list:

                    new_mask = self._mask_graph_new(mask, rate) 

                    h_m = self.model_to_explain[0](sg.to('cuda'),'author',edge_weight=new_mask)

                    src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                    dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                    all_h = torch.cat((src_h, dst_h),dim=1).to('cuda')
                    pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                    all_pred.append(pred.squeeze().detach().cpu().numpy())

            for n in edge_list:
                n = int(n)
                target_list.append(torch.round(target2[n+num_edges]).detach().cpu().numpy())

                src, dst = self.G_test_label[i][1].edges()[0][n], self.G_test_label[i][1].edges()[1][n]

                sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'author': (src,dst)}, k=2, store_ids=True)

                input_expl = self._create_explainer_input(sg, embed, src, dst).unsqueeze(0).to('cuda')

                sampling_weights = self.explainer_model(input_expl)

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to('cuda')

                for rate in rate_list:

                    new_mask = self._mask_graph_new(mask, rate) 

                    h_m = self.model_to_explain[0](sg.to('cuda'),'author',edge_weight=new_mask)

                    src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                    dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                    all_h = torch.cat((src_h, dst_h),dim=1).to('cuda')
                    pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                    all_pred.append(pred.squeeze().detach().cpu().numpy())

        for j in range(len(rate_list)):
            print(rate_list[j])
            preds = all_pred[j::len(rate_list)]
            auc = roc_auc_score(target_list, preds)
            ap = average_precision_score(target_list, preds)
            writer.add_scalar(f'loss/auc_{rate_list[j]}', auc, e)
            writer.add_scalar(f'loss/ap_{rate_list[j]}', ap, e)
            print('auc',auc)
            print('ap',ap)
