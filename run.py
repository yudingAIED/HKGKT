# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:06:26 2023

@author: wsxok
"""

from config import *
from process import *
from HKGKT import *

data_process(file, edge_index, edge_type, max_length)

Q_info = torch.from_numpy(np.load('Q_info.npy')).long().cuda()
edge_index = torch.from_numpy(np.load('edge_index.npy')).long().cuda() 
edge_type = torch.from_numpy(np.load('edge_type.npy')).long().cuda()
q = torch.from_numpy(np.load('q.npy')).long()
y = torch.from_numpy(np.load('a.npy')).long()
diff = torch.from_numpy(np.load('diff.npy')).long().cuda()

train_q, valid_q, train_y, valid_y =  train_test_split(q, y, test_size=0.2, random_state=1234)

model = KT(concept_num, concept_dim1, concept_dim2, concept_dim3, num_relations,
           diff_num, diff_dim, answer_num, answer_dim,
           output_dim1, output_dim2,
           hidden_dim, layer_dim).cuda()
model.embedding_diff.weight.data[-1] = torch.zeros(diff_dim)
model.embedding_answer.weight.data[-1] = torch.zeros(answer_dim)

train_data = Data.TensorDataset(train_q, train_y)
valid_data = Data.TensorDataset(valid_q, valid_y)
train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, 
                              shuffle = True)
valid_loader = Data.DataLoader(dataset = valid_data, batch_size = batch_size*4, 
                              shuffle = True)

optimizer = Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
L1 = torch.nn.L1Loss()
L2 = torch.nn.MSELoss()

train_loss_all = []
train_acc_all = []
train_auc_all = []

#Train
for epoch in range(num_epochs):
    print('-'*50)
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    train_truth_empty = torch.tensor([-1])
    train_pred_y_empty = torch.tensor([-1])
    train_loss = 0.0
    train_num = 0
    
    start = time.time()
    model.train()
    for step, (q,y) in enumerate(train_loader):
        q = q.cuda()
        y = y.cuda()
        pred_y, e = model(Q_info, edge_index, edge_type, q, y, diff, True)
        
        pred_y1 = pred_y
        truth_1 = y[:,1:].float()
        pred_y1 = pred_y1[truth_1<=1]
        truth_1 = truth_1[truth_1<=1]
        
        loss = loss_func(pred_y1, truth_1) + \
               k1 * L1(e[:,1:,:],e[:,:-1,:])/(concept_num-1) + \
               k2 * L2(e[:,1:,:],e[:,:-1,:])/(concept_num-1) + \
               k_p * L1(e[:,:,edge_index[:,edge_type==1][0]],e[:,:,edge_index[:,edge_type==1][1]])/(concept_num-1) + \
               k_p * (e[:,:,edge_index[:,edge_type==0][1]]-e[:,:,edge_index[:,edge_type==0][0]]).mean()/(concept_num-1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*truth_1.shape[0]        
        train_num += truth_1.shape[0]
        train_truth_empty = torch.cat((train_truth_empty,truth_1.cpu()))
        train_pred_y_empty = torch.cat((train_pred_y_empty,pred_y1.cpu()))
    train_truth_empty = train_truth_empty.numpy()
    train_pred_y_empty = train_pred_y_empty.detach().numpy()
    train_loss_all.append(train_loss/train_num)    
    train_auc_all.append(roc_auc_score(train_truth_empty[1:], train_pred_y_empty[1:]))
    train_pred_y_empty[train_pred_y_empty<=0.5]=0
    train_pred_y_empty[train_pred_y_empty>0.5]=1
    train_acc_all.append(accuracy_score(train_truth_empty[1:], train_pred_y_empty[1:]))
    end = time.time()
    print('{} Train Loss: {: .4f} Train Acc: {: .4f} Train AUC: {: .4f} time consuming: {: .4f}秒'.format(
    epoch, train_loss_all[-1], train_acc_all[-1], train_auc_all[-1], end-start))
print("Training Finish!\n")

valid_truth_empty = torch.tensor([-1])
valid_pred_y_empty = torch.tensor([-1])
valid_loss = 0.0
valid_num = 0

start = time.time()
model = model.cpu()
Q_info = Q_info.cpu()
edge_index = edge_index.cpu()
edge_type = edge_type.cpu()
diff = diff.cpu()

#valid
for step, (q,y) in enumerate(valid_loader):
    pred_y,e = model(Q_info, edge_index, edge_type, q, y, diff, False)
    
    pred_y1 = pred_y
    truth_1 = y[:,1:].float()
    pred_y1 = pred_y1[truth_1<=1]    #排除空值
    truth_1 = truth_1[truth_1<=1]    #排除空值
          
    loss = loss_func(pred_y1, truth_1) + \
           k1 * L1(e[:,1:,:],e[:,:-1,:])/(concept_num-1) + \
           k2 * L2(e[:,1:,:],e[:,:-1,:])/(concept_num-1) + \
           k_p * L1(e[:,:,edge_index[:,edge_type==1][0]],e[:,:,edge_index[:,edge_type==1][1]])/(concept_num-1) + \
           k_p * (e[:,:,edge_index[:,edge_type==0][1]]-e[:,:,edge_index[:,edge_type==0][0]]).mean()/(concept_num-1)
    
    valid_loss += loss.item()*truth_1.shape[0]        
    valid_num += truth_1.shape[0]
    valid_truth_empty = torch.cat((valid_truth_empty,truth_1.cpu()))
    valid_pred_y_empty = torch.cat((valid_pred_y_empty,pred_y1.cpu()))
valid_truth_empty = valid_truth_empty.numpy()
valid_pred_y_empty = valid_pred_y_empty.detach().numpy()
loss = valid_loss/valid_num    
auc = roc_auc_score(valid_truth_empty[1:], valid_pred_y_empty[1:])
valid_pred_y_empty[valid_pred_y_empty<=0.5]=0
valid_pred_y_empty[valid_pred_y_empty>0.5]=1
acc = accuracy_score(valid_truth_empty[1:], valid_pred_y_empty[1:])
end = time.time()
print('Valid Loss: {: .4f} Valid Acc: {: .4f} Valid AUC: {: .4f} time consuming: {: .4f}秒'.format(
    loss, acc, auc, end-start))
print("Validing Finish!\n")





