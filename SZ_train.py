import math
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.model_selection import ShuffleSplit

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_SZ_data(path='./mock_data/SZ_data/final_input/'):
    print("Loading shenzhen dataset...")

    # Load Features - FloatTensor - torch.Size([12345,dim])
    df_zonevec = pd.read_csv(path+'df_zonevec.csv')
    zonevec = df_zonevec.values[:,2:]
    features = torch.FloatTensor(zonevec)

    # Load Labels - FloatTensor - torch.Size([12345,8])
    df_area = pd.read_csv(path+'df_area.csv')
    labels = df_area.values[:,2:]
    labels = torch.FloatTensor(labels)

    # Load Spatial Interaction (Flow - ndarray (12345,12345)/Distance-decay ndarray (12345,12345)/Adjacency ndarray (12345,12345))
    flow = np.load(path+'flow_mx.npy')
    dis = np.load(path+'wdis_normalize.npy')
    adj = np.load(path+'adj.npy')

    return features,labels,flow,dis,adj

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self,input,adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class HGCNadj(nn.Module):
    def __init__(self,nfeat,nout,dropout):
        super(HGCNadj,self).__init__()
        self.gc_out = GraphConvolution(nfeat,16)
        self.gc_in = GraphConvolution(nfeat,16)
        self.dropout = dropout
        self.gc_self = GraphConvolution(nfeat, 32)
        self.layer1 = Parameter(torch.FloatTensor(64, 32))
        self.bias1 = Parameter(torch.FloatTensor(32))
        self.layer2 = Parameter(torch.FloatTensor(32, 16))
        self.bias2 = Parameter(torch.FloatTensor(16))
        self.layer3 = Parameter(torch.FloatTensor(16, nout))
        self.bias3 = Parameter(torch.FloatTensor(nout))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.layer1.size(1))
        self.layer1.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer2.size(1))
        self.layer2.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.layer3.size(1))
        self.layer3.data.uniform_(-stdv, stdv)
        self.bias3.data.uniform_(-stdv, stdv)
    
    def forward(self,x,mx,adj):
        out_feat = F.relu(self.gc_out(x,mx))
        in_feat = F.relu(self.gc_in(x,mx.T))
        x = F.relu(self.gc_self(x,adj))
        x = torch.cat([out_feat,in_feat,x],1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer1)
        x = x + self.bias1
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer2)
        x = x + self.bias2
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x,self.layer3)
        x = x + self.bias3
        return x

def train_HGCNadj(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index,total_loss,total_mae,total_cos):
    mae_loss_fn = torch.nn.L1Loss()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,mx,adj)
    loss_train = loss_fn(F.log_softmax(output[train_index],dim=1),labels[train_index])
    mae_train = mae_loss_fn(F.softmax(output[train_index],dim=1),labels[train_index])
    cos_train = cos(F.softmax(output[train_index],dim=1),labels[train_index]).mean()
    loss_train.backward()
    optimizer.step()

    # Test
    model.eval()
    output = model(features,mx,adj)
    loss_test = loss_fn(F.log_softmax(output[test_index],dim=1),labels[test_index])
    mae_test = mae_loss_fn(F.softmax(output[test_index],dim=1),labels[test_index])
    cos_test = cos(F.softmax(output[test_index],dim=1),labels[test_index]).mean()
    if loss_test.item()<total_loss:
        total_loss = loss_test.item()
        total_mae = mae_test.item()
        total_cos = cos_test.item()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train(KL_Div): {:.4f}'.format(loss_train.item()),
          'MAE_train: {:.4f}'.format(mae_train.item()),
          'COS_train: {:.4f}'.format(cos_train.item()),
          'loss_test(KL_Div): {:.4f}'.format(loss_test.item()),
          'MAE_test: {:.4f}'.format(mae_test.item()),
          'COS_test: {:.4f}'.format(cos_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return total_loss, total_mae, total_cos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--k-fold', type=int, help='K-fold validation.')
    parser.add_argument('--train_size', type=float, help='Train size.')
    args = parser.parse_args()

    # Enable cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    features, labels, flow, dis, adj = load_SZ_data()

    # Mode setting
    np.fill_diagonal(flow,0.0)
    flow = normalize(flow)
    mx = torch.FloatTensor(flow)
    adj = adj + np.eye(adj.shape[0])
    adj = normalize(adj)
    adj = torch.FloatTensor(adj)

    # K-fold split
    shuffle_folds = ShuffleSplit(n_splits=args.k_fold,train_size=args.train_size,random_state=args.seed).split(range(features.shape[0]))
    avg_loss = []
    avg_cos = []
    avg_mae = []
    for i, (train_index,test_index) in enumerate(shuffle_folds):
        print("------Fold {}:".format(i+1))
        train_index = torch.LongTensor(train_index)
        test_index = torch.LongTensor(test_index)
        
        # Model and optimizer
        model = HGCNadj(nfeat=features.shape[1],nout=labels.shape[1],dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        
        # Cuda settings
        if args.cuda:
            model.cuda()
            features = features.cuda()
            labels = labels.cuda()
            train_index = train_index.cuda()
            test_index = test_index.cuda()
            mx = mx.cuda()
            adj = adj.cuda()

        # Train model
        total_loss = 10
        total_mae = 10
        total_cos = 10
        t_total = time.time()
        for epoch in range(args.epochs):
            total_loss, total_mae, total_cos = train_HGCNadj(epoch,model,optimizer,loss_fn,features,labels,mx,adj,train_index,test_index,total_loss,total_mae,total_cos)
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("Best Loss:{}".format(total_loss))
        avg_loss.append(total_loss)
        avg_mae.append(total_mae)
        avg_cos.append(total_cos)
    avg_loss = np.array(avg_loss)
    avg_mae = np.array(avg_mae)
    avg_cos = np.array(avg_cos)
    print("Average Loss:{:.4f}".format(np.mean(avg_loss)))
    print("Average MAE:{:.4f}".format(np.mean(avg_mae)))
    print("Average COS:{:.4f}".format(np.mean(avg_cos)))
    print("Std Loss:{:.4f}".format(np.std(avg_loss)))
    print("Std MAE:{:.4f}".format(np.std(avg_mae)))
    print("Std COS:{:.4f}".format(np.std(avg_cos)))