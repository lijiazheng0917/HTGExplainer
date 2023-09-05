import torch
from torch import nn
from torch.optim import Adam, AdamW
from tqdm import tqdm
import dgl
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import random
from random import sample
from utils.pytorchtools import EarlyStopping
from utils.utils import compute_metric, TimeEncode
from model.BaseExplainer import HTGNNExplainer

from torch.utils.tensorboard import SummaryWriter 
# writer = SummaryWriter('output1/ml_k1_v2')

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
    def __init__(self, model_to_explain, G_train, G_train_label, G_val, G_val_label, G_test, G_test_label, time_win, node_emb, device,
                epochs, lr, warmup_epoch, es_epoch, explain_method, batch_size, khop, te,he,test_only,
                temp=(5.0, 2.0), reg_coefs=(1e-4, 1e-2),sample_bias=0):
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
        self.node_emb = node_emb
        self.device = device
        self.warmup_epoch = warmup_epoch
        self.es_epoch = es_epoch
        self.explain_method = explain_method
        self.bs = batch_size
        self.khop = khop
        self.te = te
        self.he = he
        self.test_only = test_only
        if self.explain_method == 'our':
            self.writer = SummaryWriter('output1/ml_'+self.te+self.he)
        else:
            self.writer = SummaryWriter('output1/ml_pg')

        # self.expl_embedding = self.node_emb * 4 +32+32


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
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(self.device)
            
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


    def _mask_graph_new(self, mask, rate):
        
        new_mask = torch.zeros(mask.shape[0]).to('cuda')
        _, idx = torch.sort(mask,descending=True)
        top_idx = idx[:int(rate*len(idx))]
        new_mask[top_idx]=1

        return new_mask
        # new_mask = mask.clone()
        # _, idx = torch.sort(mask,descending=True)
        # bottom_idx = idx[int(rate*len(idx)):]
        # new_mask[bottom_idx] = 0
        # return new_mask

    def evaluate(self, exp, G_val, G_val_label, t):
        exp.eval()
        loss = 0
        for i in range(len(G_val)):

            feat = self.model_to_explain[0](self.G_val[i].to(self.device))
            h_u, h_m = feat['user'], feat['movie']

            embed = feat
            embed['user'] = embed['user'].detach()
            embed['movie'] = embed['movie'].detach()

            pos_u, pos_m = self.G_val_label[i][0][0], self.G_val_label[i][0][1]
            neg_u, neg_m = self.G_val_label[i][1][0], self.G_val_label[i][1][1]

            pos_score, neg_score = self.model_to_explain[1](h_u, h_m, pos_u, pos_m, neg_u, neg_m)

            ori_pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
            target2 = torch.sigmoid(ori_pred).detach()


            num_edges = len(self.G_val_label[i][0][0])
            edge_list = list(range(num_edges))
            # how many edges to test?
            # edge_list = sample(edge_list, 1000)
            edge_list = edge_list[:300]
                
            for n in edge_list:
                n = int(n)

                flag = random.choice([0,1])
                src, dst = G_val_label[i][flag][0][n].to(self.device), G_val_label[i][flag][1][n].to(self.device)

                sg, _ = dgl.khop_in_subgraph(G_val[i], {'user': src,'movie':dst}, k=self.khop, store_ids=True)

                #input_expl = self._create_explainer_input(sg, embed, src, dst, self.het_emb, self.etype_dic, self.tem_emb).unsqueeze(0).to(self.device)

                #sampling_weights = self.explainer_model(input_expl)
                sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device, self.explain_method,'ml')

                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to(self.device)

                h_m = self.model_to_explain[0](sg.to(self.device),edge_weight=mask)
                h_m_u, h_m_m = h_m['user'], h_m['movie']

                src_h = h_m_u[torch.where(sg.ndata[dgl.NID]['user'] == src)]
                dst_h = h_m_m[torch.where(sg.ndata[dgl.NID]['movie'] == dst)]

                all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                
                if flag == 0:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n])
                else:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n+num_edges])

                loss_ = cce_loss.item()
                size_reg = self.reg_coefs[0]
                size_loss_ = torch.sum(mask) * size_reg

                loss = loss + loss_ + size_loss_.item()     

                loss += loss_      
                
        return loss

    def explain(self):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set

        # het embedding
        elist = []
        for srctype, etype, dsttype in self.G_train[0].canonical_etypes:
            elist.append(etype[:-3])
        num_types = len(set(elist))
        self.etype_dic = {edge_type: i for i, edge_type in enumerate(set(elist))}

        te = self.te
        he = self.he
        explain_method = self.explain_method
        node_emb = self.node_emb
        tw =self.tw

    
        self.exp = HTGNNExplainer(te, he, explain_method, num_types, node_emb, tw)
        
        # self.t_linear = 

        self.train()

    def train(self):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        # self.explainer_model = self.explainer_model.to(self.device)
        self.exp = self.exp.to(self.device)
        #print(self.exp.tem_emb.device)
        # self.het_emb = self.het_emb.to(self.device)
        # self.tem_emb = self.tem_emb.to(self.device)

        # Create optimizer and temperature schedule
        # optimizer = Adam([*self.explainer_model.parameters(),*self.het_emb.parameters()], lr=self.lr, weight_decay=1e-4)
        optimizer = Adam(self.exp.parameters(), lr=self.lr, weight_decay=1e-4)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        model_out_path = 'output1/explainer_ml_seed2024'
        # early_stopping
        early_stopping = EarlyStopping(patience=self.es_epoch, verbose=True, path=f'{model_out_path}/checkpoint_ml_{self.explain_method}_{self.te}_{self.he}.pt')

        size_reg = self.reg_coefs[0]
        entropy_reg = self.reg_coefs[1]

        #rate_list = [0.001,0.003,0.005,0.01,0.03,0.05]
        rate_list = [0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,0.7,0.9]
        # [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7]

        # Start training loop
        for e in tqdm(range(0, self.epochs)):

            # print(self.exp.het_emb)

            if self.test_only: break

            # self.explainer_model.train()
            self.exp.train()
            t = temp_schedule(e)
            epoch_loss = 0
            epoch_closs = 0
            epoch_mloss = 0
            epoch_sloss = 0

            for i in range(len(self.G_train)):
            # for i in range(1):
                self.G_train[i] = self.G_train[i].to(self.device)

                feat = self.model_to_explain[0](self.G_train[i])
                h_u, h_m = feat['user'], feat['movie']

                pos_u, pos_m = self.G_train_label[i][0][0], self.G_train_label[i][0][1]
                neg_u, neg_m = self.G_train_label[i][1][0], self.G_train_label[i][1][1]
                pos_score, neg_score = self.model_to_explain[1](h_u, h_m, pos_u, pos_m, neg_u, neg_m)

                ori_pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
                target = torch.sigmoid(ori_pred).detach()

                embed = feat
                embed['user'] = embed['user'].detach()
                embed['movie'] = embed['movie'].detach()

                num = len(self.G_train_label[i][0][0])
                all_loss = 0
                all_closs = 0
                all_mloss = 0
                all_sloss = 0

                # TODO change here
                for j in range(5):
                    optimizer.zero_grad()
                    loss = 0
                    closs = 0
                    mloss = 0
                    sloss = 0

                    # lo = 16*j
                    # hi = min(16*(j+1),self.G_train_label[i][0].num_edges())
                    # hi = 16*(j+1)
                    # print(lo,hi)
                    training_list = sample(list(range(num)),self.bs)
                    # for n in training_list:  # 0-8,8-16,...
                    for n in training_list:
                        n = int(n)
                        flag = random.choice([0,1])

                        src, dst = self.G_train_label[i][flag][0][n].to(self.device), self.G_train_label[i][flag][1][n].to(self.device)

                        sg, _ = dgl.khop_in_subgraph(self.G_train[i], {'user': src,'movie':dst}, k=self.khop, store_ids=True)

                        #input_expl = self._create_explainer_input(sg, embed, src, dst, self.het_emb, self.etype_dic, self.tem_emb).unsqueeze(0).to(self.device)

                        #sampling_weights = self.explainer_model(input_expl)

                        sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device, self.explain_method,'ml')

                        mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to(self.device)

                        h_m = self.model_to_explain[0](sg.to(self.device),edge_weight=mask)
                        h_m_u, h_m_m = h_m['user'], h_m['movie']

                        src_h = h_m_u[torch.where(sg.ndata[dgl.NID]['user'] == src)]
                        dst_h = h_m_m[torch.where(sg.ndata[dgl.NID]['movie'] == dst)]

                        all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                        pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                        
                        if flag == 0:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n])
                        else:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n+num])

                        size_loss = torch.sum(mask) * size_reg
                        mask_ent_reg1 = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
                        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg1)
                        loss_ = cce_loss + size_loss# + mask_ent_loss
                        
                        loss += loss_
                        closs += cce_loss.item()
                        sloss += size_loss.item()
                        mloss += mask_ent_loss.item() 

                    loss.backward()
                    optimizer.step()

                    all_loss += loss.item()
                    all_closs += closs
                    all_mloss += mloss
                    all_sloss += sloss
            
                epoch_loss += all_loss
                epoch_closs += all_closs
                epoch_mloss += all_mloss
                epoch_sloss += all_sloss
            # print(epoch_closs)
            self.writer.add_scalar('loss/all_loss', epoch_loss, e)
            self.writer.add_scalar('loss/closs', epoch_closs, e)
            self.writer.add_scalar('loss/mloss', epoch_mloss, e)
            self.writer.add_scalar('loss/sloss', epoch_sloss, e)

            torch.cuda.empty_cache()
        
            # if e >=0:
            #     eval_loss = self.evaluate(self.explainer_model, self.G_val, self.G_val_label, t)
            #     torch.save(self.explainer_model.state_dict(), f'{model_out_path}/checkpoint_ml_{self.explain_method}_{self.te}_{self.he}.pt')
            #     break
            if e>self.warmup_epoch:
            # if e>0:
                eval_loss = self.evaluate(self.exp, self.G_val, self.G_val_label, t)
                early_stopping(eval_loss, self.exp)
                if early_stopping.early_stop:
                    print("Early stopping", e)
                    break
            
        # testing
        if self.test_only:
            for test_ in [0]:

                print(test_)

                # self.explainer_model.load_state_dict(torch.load(f'{model_out_path}/checkpoint_ml_{self.explain_method}_{self.te}_{self.he}.pt'))
                #model_out_path = f'output1/explainers_{test_}/ml'
                model_out_path = f'output1/explainer_ml_seed2024'
                self.exp.load_state_dict(torch.load(f'{model_out_path}/checkpoint_ml_{self.explain_method}_{self.te}_{self.he}.pt'))
                self.exp.eval()
                all_pred = []
                target_list = []
                for i in range(len(self.G_test)):
                    feat = self.model_to_explain[0](self.G_test[i].to(self.device))
                    embed = feat
                    embed['user'] = embed['user'].detach()
                    embed['movie'] = embed['movie'].detach()

                    h_u, h_m = feat['user'], feat['movie']

                    pos_u, pos_m = self.G_test_label[i][0][0], self.G_test_label[i][0][1]
                    neg_u, neg_m = self.G_test_label[i][1][0], self.G_test_label[i][1][1]

                    pos_score, neg_score = self.model_to_explain[1](h_u, h_m, pos_u, pos_m, neg_u, neg_m)

                    ori_pred2 = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
                    target2 = torch.sigmoid(ori_pred2).detach()

                    num_edges = len(self.G_test_label[i][0][0])
                    edge_list = list(range(num_edges))
                    # how many edges to test?
                    # edge_list = sample(edge_list, 1000)
                    edge_list = edge_list[:1000]
        
                    for n in edge_list:
                        n = int(n)
                        #print(i)
                        target_list.append(torch.round(target2[n]).detach().cpu().numpy())

                        src, dst = self.G_test_label[i][0][0][n].to(self.device), self.G_test_label[i][0][1][n].to(self.device)
                        #print(src,dst)

                        sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'user': src,'movie':dst}, k=self.khop, store_ids=True)

                        #input_expl = self._create_explainer_input(sg, embed, src, dst, self.het_emb, self.etype_dic, self.tem_emb).unsqueeze(0).to(self.device)

                        #sampling_weights = self.explainer_model(input_expl)
                        sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device, self.explain_method,'ml')

                        mask = self._sample_graph(sampling_weights, bias=self.sample_bias, training=False).squeeze().to(self.device)
        
                        #print(src,dst)
                        #torch.save(mask,'output1/case.pt')
                        #break
                    #break


                        for rate in rate_list:

                            new_mask = self._mask_graph_new(mask, rate) 

                            h_m = self.model_to_explain[0](sg.to(self.device),edge_weight=new_mask)
                            h_m_u, h_m_m = h_m['user'], h_m['movie']

                            src_h = h_m_u[torch.where(sg.ndata[dgl.NID]['user'] == src)]
                            dst_h = h_m_m[torch.where(sg.ndata[dgl.NID]['movie'] == dst)]

                            all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                            pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                            all_pred.append(pred.squeeze().detach().cpu().numpy())

                    for n in edge_list:
                        n = int(n)
                        target_list.append(torch.round(target2[n+num_edges]).detach().cpu().numpy())

                        src, dst = self.G_test_label[i][1][0][n].to(self.device), self.G_test_label[i][1][1][n].to(self.device)

                        sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'user': src,'movie':dst}, k=self.khop, store_ids=True)

                        #input_expl = self._create_explainer_input(sg, embed, src, dst, self.het_emb, self.etype_dic, self.tem_emb).unsqueeze(0).to(self.device)

                        #sampling_weights = self.explainer_model(input_expl)
                        sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device, self.explain_method,'ml')

                        mask = self._sample_graph(sampling_weights, bias=self.sample_bias, training=False).squeeze().to(self.device)

                        for rate in rate_list:

                            new_mask = self._mask_graph_new(mask, rate) 

                            h_m = self.model_to_explain[0](sg.to(self.device),edge_weight=new_mask)
                            h_m_u, h_m_m = h_m['user'], h_m['movie']

                            src_h = h_m_u[torch.where(sg.ndata[dgl.NID]['user'] == src)]
                            dst_h = h_m_m[torch.where(sg.ndata[dgl.NID]['movie'] == dst)]

                            all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                            pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                            all_pred.append(pred.squeeze().detach().cpu().numpy())

                auc_list = []
                ap_list = []
                for j in range(len(rate_list)):
                    #print(rate_list[j])
                    preds = all_pred[j::len(rate_list)]
                    auc = roc_auc_score(target_list, preds)
                    ap = average_precision_score(target_list, preds)
                    # writer.add_scalar(f'loss/auc_{rate_list[j]}', auc, e)
                    # writer.add_scalar(f'loss/ap_{rate_list[j]}', ap, e)
                    auc_list.append(auc)
                    ap_list.append(ap)
                    #print('auc',auc)
                    #print('ap',ap)
                print(rate_list)
                print('auc',auc_list)
                print('ap',ap_list)
