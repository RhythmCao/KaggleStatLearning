#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNNModel(nn.Module):

    def __init__(self, infeats, outfeats, layers=[], dropout=0.5,\
                    nonlinear='relu', batchnorm=False, device=None):
        super(FNNModel,self).__init__()
        assert type(layers)==list
        self.num_layers=len(layers)
        self.dropout_layer=nn.Dropout(p=dropout)
        self.use_batchnorm = batchnorm
        self.nonlinear=nonlinear.lower() 
        assert self.nonlinear in ['relu','sigmoid','tanh']
        linear_layer=[0]*(self.num_layers+1)
        if self.use_batchnorm:
            batchnorm_layer=[0]*self.num_layers
        for idx, hidden_size in enumerate(layers):
            linear_layer[idx]=nn.Linear(infeats, layers[idx])
            if self.use_batchnorm:
                batchnorm_layer[idx]=nn.BatchNorm1d(layers[idx])
            infeats=layers[idx]
        linear_layer[self.num_layers]=nn.Linear(infeats, outfeats)
        self.linear_layer=nn.ModuleList(linear_layer)
        if self.use_batchnorm:
            self.batchnorm_layer=nn.ModuleList(batchnorm_layer)
        self.device = device

    def init_weight(self, weight_scale=0.2):
        for each_layer in self.linear_layer:
            torch.nn.init.xavier_uniform_(each_layer.weight, weight_scale)
            torch.nn.init.constant_(each_layer.bias, 0)
        if self.use_batchnorm:
            for each_layer in self.batchnorm_layer:
                torch.nn.init.xavier_uniform_(each_layer.weight, weight_scale)
                torch.nn.init.constant_(each_layer.bias, 0)

    def forward(self, inputs):
        infeats = inputs
        for idx in range(self.num_layers):
            outfeats=self.linear_layer[idx](self.dropout_layer(infeats))
            if self.use_batchnorm:
                outfeats=self.batchnorm_layer[idx](outfeats)
            if self.nonlinear=='relu':
                infeats=F.relu(outfeats) 
            elif self.nonlinear=='sigmoid':
                infeats=F.sigmoid(outfeats)
            elif self.nonlinear=='tanh':
                infeats=F.tanh(outfeats)
            else:
                infeats = outfeats # no non-linear activation
        scores = self.linear_layer[self.num_layers](self.dropout_layer(infeats))
        scores = F.log_softmax(scores, dim=-1) # batch_size, num_classes
        return scores

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb')) 
