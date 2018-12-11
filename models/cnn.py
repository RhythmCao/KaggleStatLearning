#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self, width, height, outfeats, channel=[], kernel_size=[], maxpool_kernel_size=[], \
                    nonlinear='relu', batchnorm=False, dropuout=0.5, device=None):
        super(CNNModel,self).__init__()
        assert type(layers)==list
        self.num_layers = len(channel)
        self.width, self.height = width, height
        self.num_classes = outfeats
        self.nonlinear=nonlinear.lower() 
        if self.nonlinear == 'relu':
            self.nonlinear = nn.ReLU()
        elif self.nonlinear == 'sigmoid':
            self.nonlinear = nn.Sigmoid()
        elif self.nonlinear == 'tanh':
            self.nonlinear = nn.Tanh()
        else:
            self.nonlinear = nn.ReLU()
        if maxpool_kernel_size == [] or None:
            maxpool_kernel_size = [2] * self.num_layers
        self.use_batchnorm = batchnorm
        conv_layer = [0]*(self.num_layers)
        in_channel, width = 1, self.width
        for idx in range(self.num_layers):
            assert kernel_size[idx] % 2 == 1
            # seldom use dropout in Conv Layer
            conv_layer[idx] = nn.Sequential(
                nn.Conv2d(in_channel, channel[idx], kernel_size[idx], padding=(kernel_size[idx]-1)/2, stride=1, dilation=1),
                nn.BatchNorm2d(channel[idx]),
                self.nonlinear,
                nn.MaxPool2d(maxpool_kernel_size[idx])
            ) if self.use_batchnorm else \
            nn.Sequential(
                nn.Conv2d(in_channel, channel[idx], kernel_size[idx], padding=(kernel_size[idx]-1)/2, stride=1, dilation=1),
                self.nonlinear,
                nn.MaxPool2d(maxpool_kernel_size[idx])
            )
            in_channel = channel[idx]
            width = width/maxpool_kernel_size[idx]
        self.conv_layer = nn.ModuleList(conv_layer)
        self.out_layer = nn.Linear(in_channel*width*width, self.num_classes)
        self.dropout_layer = nn.Dropout(p=dropuout)
        self.device = device

    def init_weight(self, weight_scale=0.2):
        for each_layer in self.conv_layer:
            torch.nn.init.xavier_uniform_(each_layer[0].weight, weight_scale)
            torch.nn.init.constant_(each_layer[0].bias, 0)
            if self.use_batchnorm:
                torch.nn.init.xavier_uniform_(each_layer[1].weight, weight_scale)
                torch.nn.init.constant_(each_layer[1].bias, 0)
        torch.nn.init.xavier_uniform_(self.out_layer.weight, weight_scale)
        torch.nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, inputs):
        infeats = inputs.contiguous().view(inputs.shape[0], 1, self.height, self.width)
        for idx in range(self.num_layers):
            outfeats = self.conv_layer[idx](infeats)
            infeats = outfeats
        scores = self.output_layer(self.dropout_layer(outfeats.view(outfeats.size(0), -1)))
        scores = F.log_softmax(scores, dim=-1) # batch_size, num_classes
        return scores

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb')) 
