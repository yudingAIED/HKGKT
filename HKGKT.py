# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:58:40 2023

@author: wsxok
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from tqdm import tqdm

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import RGCNConv

class KT(nn.Module):
    def __init__(self, concept_num, concept_dim1, concept_dim2, concept_dim3, num_relations,
                 diff_num, diff_dim, answer_num, answer_dim,
                 output_dim1, output_dim2,
                 hidden_dim, layer_dim):
        super(KT, self).__init__()
        self.concept_num    = concept_num
        self.concept_dim1   = concept_dim1
        self.concept_dim2   = concept_dim2
        self.concept_dim3   = concept_dim3
        self.num_relations  = num_relations
        self.diff_num       = diff_num
        self.diff_dim       = diff_dim
        self.answer_num     = answer_num
        self.answer_dim     = answer_dim
            
        self.output_dim1    = output_dim1
        self.output_dim2    = output_dim2
            
        self.hidden_dim     = hidden_dim
        self.layer_dim      = layer_dim
            
        #self.conv1 = RGCNConv(self.concept_dim1, self.concept_dim2, self.num_relations)
        self.conv2 = RGCNConv(self.concept_dim2, self.concept_dim3, self.num_relations)
            
        self.embedding_diff = nn.Embedding(self.diff_num, self.diff_dim)
        
        self.embedding_answer = nn.Embedding(self.answer_num, self.answer_dim)
            
        self.fc1 = nn.Linear(self.concept_dim3 + self.diff_dim, self.output_dim1)

        self.fc2 = nn.Linear(self.output_dim1 + self.answer_dim, self.output_dim2)

        self.lstm0 = nn.LSTM(self.output_dim2, hidden_dim, layer_dim) 

        self.fc3 = nn.Linear(self.hidden_dim, self.concept_num)
        
        self.active1 = nn.ReLU()
        self.active2 = nn.Sigmoid()
    
    def forward(self, Q_info, edge_index, edge_type, q, y, diff, device):
        
        if device == True:
            c = torch.arange(start=0, end=self.concept_num).cuda()
        else:
            c = torch.arange(start=0, end=self.concept_num)

        #c = self.conv1(c, edge_index, edge_type)
        #c = self.active1(c)
        c = self.conv2(c, edge_index, edge_type)

        concept = c[Q_info[q]]
        
        difficulty = self.embedding_diff(diff[q])
        
        text = torch.cat((concept,difficulty),dim=-1)
        text = self.fc1(text)
        
        answer = self.embedding_answer(y)
        
        X = torch.cat((text,answer),dim=-1)
        X = self.fc2(X)
        out, (h_n, h_c) = self.lstm0(X, None)
        
        e = self.fc3(out)
        e = self.active2(e)  
        index = Q_info[q[:,1:]]
        if device:
            index = torch.zeros(index.shape[0],index.shape[1],
                                self.concept_num).cuda().scatter_(2,index.unsqueeze(2),1)
        else:
            index = torch.zeros(index.shape[0],index.shape[1],
                                self.concept_num).scatter_(2,index.unsqueeze(2),1)
        res = torch.sum(e[:,:-1,:]*index,dim = -1)
        res = self.active2(res-(diff[q][:,1:]*0.2+0.2))
        
        return res,e

