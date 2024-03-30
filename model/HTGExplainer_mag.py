import os
import torch
from torch.optim import Adam
from tqdm import tqdm
import dgl
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from random import sample
from utils.pytorchtools import EarlyStopping
from model.BaseExplainer import HTGNNExplainer

from torch.utils.tensorboard import SummaryWriter 


class HTGExplainer():
    """
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
                epochs, lr, warmup_epoch, es_epoch, batch_size, khop,te,he,test_only,
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
        self.bs = batch_size
        self.khop = khop
        self.te = te
        self.he = he
        self.test_only = test_only
        # self.writer = SummaryWriter('output1/mag_'+self.te+self.he)


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
            bias = bias + 0.0001  # bias cannot be 0
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
        """
        Mask the graph based on the given rate.
        :param mask: the mask to be applied to the graph
        :param rate: the rate at which we want to mask the graph
        :return: masked graph
        """
        
        new_mask = torch.zeros(mask.shape[0]).to(self.device)
        _, idx = torch.sort(mask,descending=True)
        top_idx = idx[:int(rate*len(idx))]
        new_mask[top_idx]=1

        return new_mask


    def evaluate(self, exp, G_val, G_val_label, t):
        exp.eval()
        loss = 0
        for i in range(len(G_val)):
            embed = {}
            for ntype in G_val[i].ntypes:
                embed[ntype] = self.model_to_explain[0](G_val[i].to(self.device), ntype).detach()

            org_feat2 = self.model_to_explain[0](self.G_val[i].to(self.device),'author')
            pos_label2, neg_label2 = G_val_label[i][0].to(self.device), G_val_label[i][1].to(self.device)
            pos_score2 = self.model_to_explain[1](pos_label2, org_feat2)
            neg_score2 = self.model_to_explain[1](neg_label2, org_feat2)
            ori_pred2 = torch.cat((pos_score2.squeeze(1), neg_score2.squeeze(1)))
            target2 = torch.sigmoid(ori_pred2).detach()

            num_edges = G_val_label[i][0].num_edges()
            edge_list = list(range(num_edges))
            # how many edges to test
            edge_list = edge_list[:300]
                
            for n in edge_list:
                n = int(n)
                flag = random.choice([0,1])

                # generate the embeddings
                src, dst = G_val_label[i][flag].edges()[0][n], G_val_label[i][flag].edges()[1][n]
                sg, _ = dgl.khop_in_subgraph(G_val[i], {'author': (src,dst)}, k=self.khop, store_ids=True)
                sampling_weights = exp(sg, embed, src, dst, self.etype_dic, self.device,'mag')
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to(self.device)
                h_m = self.model_to_explain[0](sg.to(self.device),'author',edge_weight=mask)

                src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                
                if flag == 0:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n])
                else:
                    cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target2[n+num_edges])
                loss_ = cce_loss.item()
                size_reg = self.reg_coefs[0]
                size_loss_ = torch.sum(mask).item() * size_reg

                loss = loss + loss_ + size_loss_     
                
        return loss


    def explain(self):

        # create the edge type dictionary
        elist = []
        for srctype, etype, dsttype in self.G_train[0].canonical_etypes:
            elist.append(etype[:-3])
        num_types = len(set(elist))
        self.etype_dic = {edge_type: i for i, edge_type in enumerate(set(elist))}

        # create the mlp explainer model
        self.exp = HTGNNExplainer(self.te, self.he, num_types, self.node_emb, self.tw)

        self.train()


    def train(self):

        self.exp = self.exp.to(self.device)

        optimizer = Adam(self.exp.parameters(), lr=self.lr, weight_decay=1e-4)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        model_out_path = 'ckpt/explainer_mag'
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        early_stopping = EarlyStopping(patience=self.es_epoch, verbose=True, path=f'{model_out_path}/checkpoint_mag_{self.te}_{self.he}.pt')

        size_reg = self.reg_coefs[0]
        entropy_reg = self.reg_coefs[1]

        # mask rate
        rate_list = [0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,0.7,0.9]

        # Start training loop
        for e in tqdm(range(0, self.epochs)):

            if self.test_only: break

            self.exp.train()
            t = temp_schedule(e)
            epoch_loss = 0
            epoch_closs = 0
            epoch_mloss = 0
            epoch_sloss = 0

            for i in range(len(self.G_train)):
                # generate labels
                org_feat = self.model_to_explain[0](self.G_train[i].to(self.device),'author')
                pos_label, neg_label = self.G_train_label[i][0].to(self.device), self.G_train_label[i][1].to(self.device)
                pos_score = self.model_to_explain[1](pos_label, org_feat)
                neg_score = self.model_to_explain[1](neg_label, org_feat)
                ori_pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
                target = torch.sigmoid(ori_pred).detach()

                embed = {}
                for ntype in self.G_train[i].ntypes:
                    embed[ntype] = self.model_to_explain[0](self.G_train[i].to(self.device), ntype).detach()
                num = self.G_train_label[i][0].num_edges()

                all_loss = 0
                all_closs = 0
                all_mloss = 0
                all_sloss = 0

                for j in range(0, 10):
                    optimizer.zero_grad()
                    loss = 0
                    closs = 0
                    mloss = 0
                    sloss = 0

                    training_list = sample(list(range(num)),self.bs)
                    for n in training_list:

                        n = int(n)
                        flag = random.choice([0,1])

                        # generate embeddings and predicts
                        src, dst = self.G_train_label[i][flag].edges()[0][n], self.G_train_label[i][flag].edges()[1][n]
                        sg, _ = dgl.khop_in_subgraph(self.G_train[i], {'author': (src,dst)}, k=self.khop, store_ids=True)
                        sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device,'mag')
                        mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze().to(self.device)
                        h_m = self.model_to_explain[0](sg.to(self.device),'author',edge_weight=mask)

                        src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                        dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                        all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                        pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))
                        
                        # compute losses
                        if flag == 0:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n])
                        else:
                            cce_loss = F.binary_cross_entropy_with_logits(pred.squeeze(),target[n+num])

                        size_loss = torch.sum(mask) * size_reg
                        mask_ent_reg1 = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
                        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg1)
                        loss_ = cce_loss + size_loss + mask_ent_loss
                        
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


            # self.writer.add_scalar('loss/all_loss', epoch_loss, e)
            # self.writer.add_scalar('loss/closs', epoch_closs, e)
            # self.writer.add_scalar('loss/mloss', epoch_mloss, e)
            # self.writer.add_scalar('loss/sloss', epoch_sloss, e)

            torch.cuda.empty_cache()

            # early stopping
            if e>self.warmup_epoch:
                eval_loss = self.evaluate(self.exp, self.G_val, self.G_val_label, t)
                early_stopping(eval_loss, self.exp)
                if early_stopping.early_stop:
                    print("Early stopping", e)
                    break


        # testing
        if self.test_only:
    
            # load the model
            model_out_path = f'ckpt/explainer_mag'
            self.exp.load_state_dict(torch.load(f'{model_out_path}/checkpoint_mag_{self.te}_{self.he}.pt'))
            self.exp.eval()
            
            all_pred = []
            target_list = []
            for i in range(len(self.G_test)):
                embed = {}
                for ntype in self.G_test[i].ntypes:
                    embed[ntype] = self.model_to_explain[0](self.G_test[i].to(self.device), ntype).detach()

                org_feat2 = self.model_to_explain[0](self.G_test[i].to(self.device),'author')
                pos_label2, neg_label2 = self.G_test_label[i][0].to(self.device), self.G_test_label[i][1].to(self.device)
                pos_score2 = self.model_to_explain[1](pos_label2, org_feat2)
                neg_score2 = self.model_to_explain[1](neg_label2, org_feat2)
                ori_pred2 = torch.cat((pos_score2.squeeze(1), neg_score2.squeeze(1)))
                target2 = torch.sigmoid(ori_pred2).detach()

                num_edges = self.G_test_label[i][0].num_edges()
                edge_list = list(range(num_edges))

                # how many edges to test
                edge_list = edge_list[:1000]

                for n in edge_list:
                    n = int(n)
                    target_list.append(torch.round(target2[n]).detach().cpu().numpy())

                    src, dst = self.G_test_label[i][0].edges()[0][n], self.G_test_label[i][0].edges()[1][n]
                    sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'author': (src,dst)}, k=self.khop, store_ids=True)
                    sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device,'mag')
                    mask = self._sample_graph(sampling_weights, bias=self.sample_bias, training=False).squeeze().to(self.device)

                    for rate in rate_list:

                        new_mask = self._mask_graph_new(mask, rate) 
                        h_m = self.model_to_explain[0](sg.to(self.device),'author',edge_weight=new_mask)

                        src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                        dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                        all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                        pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                        all_pred.append(pred.squeeze().detach().cpu().numpy())

                for n in edge_list:
                    n = int(n)
                    target_list.append(torch.round(target2[n+num_edges]).detach().cpu().numpy())

                    src, dst = self.G_test_label[i][1].edges()[0][n], self.G_test_label[i][1].edges()[1][n]
                    sg, _ = dgl.khop_in_subgraph(self.G_test[i], {'author': (src,dst)}, k=self.khop, store_ids=True)
                    sampling_weights = self.exp(sg, embed, src, dst, self.etype_dic, self.device,'mag')
                    mask = self._sample_graph(sampling_weights, bias=self.sample_bias, training=False).squeeze().to(self.device)

                    for rate in rate_list:

                        new_mask = self._mask_graph_new(mask, rate) 
                        h_m = self.model_to_explain[0](sg.to(self.device),'author',edge_weight=new_mask)

                        src_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == src)]
                        dst_h = h_m[torch.where(sg.ndata[dgl.NID]['author'] == dst)]

                        all_h = torch.cat((src_h, dst_h),dim=1).to(self.device)
                        pred = self.model_to_explain[1].fc2(F.relu(self.model_to_explain[1].fc1(all_h)))

                        all_pred.append(pred.squeeze().detach().cpu().numpy())

            # calculate the auc and ap
            auc_list = []
            ap_list = []
            for j in range(len(rate_list)):
                preds = all_pred[j::len(rate_list)]
                auc = roc_auc_score(target_list, preds)
                ap = average_precision_score(target_list, preds)
                auc_list.append(auc)
                ap_list.append(ap)
            print(rate_list)
            print('auc',auc_list)
            print('ap',ap_list)