# -*- coding: utf-8 -*-
"""
@author: tsharma2
"""

#import time
import random
import numpy as np
import torch
#import os
import sys
#from torchvision import datasets
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader, Dataset
from sklearn.feature_selection import SelectKBest, f_classif

#return cancer dataloader
# dataset definition
class CSVDataset():
    # load the dataset
    def __init__(self,Dataset_X,Dataset_Y):
        # store the inputs and outputs
        self.X = Dataset_X
        self.y = Dataset_Y

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
def create_target_samples_cancer(train_targetData,n=1):
    X,Y,X_val,Y_val=[],[],[],[]
    classes=2*[n]
    validation_set = None
    i=0
    while True:
        if len(X)==n*2:
            break
        else:
            x,y=train_targetData[i]
            x = torch.tensor(x)
            y = torch.tensor(y)
            if classes[y]>0:
                X.append(x)
                Y.append(y)
                classes[y]-=1
            else:
                X_val.append(x)
                Y_val.append(y)
                validation_set = True
        i+=1
        #print(i)

    assert (len(X)==n*2)
    
    if validation_set==True:
        return torch.stack(X,dim=0),torch.from_numpy(np.array(Y)),torch.stack(X_val,dim=0),torch.from_numpy(np.array(Y_val)),validation_set
    else:
        return torch.stack(X,dim=0),torch.from_numpy(np.array(Y)),X_val,Y_val,validation_set
    

"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s,Y_s,X_t,Y_t,seed=1):
    #change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n=X_t.shape[0] #10*shot

    #shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]
    #print(classes)

    class_num=classes.shape[0]
    #print(class_num)
    
    shot=n//class_num
    #print(shot)

    def s_idxs(c):
        idx=torch.nonzero(Y_s.eq(int(c)))

        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    
    def t_idxs(c):
        idx=torch.nonzero(Y_t.eq(int(c)))
        
        return idx[torch.randperm(len(idx))][:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    #print(source_idxs)
    
    target_idxs = list(map(t_idxs, classes))
    #print(target_idxs)

    source_matrix=torch.stack(source_idxs)

    target_matrix=torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]

    for i in range(2):
        # print('i is:')
        # print(i)
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%2][j]],X_s[source_matrix[(i+1)%2][j]]))
            Y3.append((Y_s[source_matrix[i % 2][j]], Y_s[source_matrix[(i + 1) % 2][j]]))
            G4.append((X_s[source_matrix[i%2][j]],X_t[target_matrix[(i+1)%2][j]]))
            Y4.append((Y_s[source_matrix[i % 2][j]], Y_t[target_matrix[(i + 1) % 2][j]]))

    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    #make sure we sampled enough samples
    for g in groups:
        assert(len(g)==n)
    return groups,groups_y

def sample_groups(X_s,Y_s,X_t,Y_t,seed=1):

    # print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,seed=seed)

def get_k_best(X_train, y_train, X_test, k=200):
    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(X_train, y_train)
    res = (k_best.transform(X_train),
           k_best.transform(X_test))
    return res

# Constrastive Semantic Alignment Loss
def csa_loss(x, y, class_eq):
    margin = 1
    dist = torch.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

# Initialization.Create_Pairs
def CCSA_create_pairs(domain_adaptation_task,
                      repetition,sample_per_class,
                      X_train_target, y_train_target,
                      X_train_source, y_train_source,
                      n_features=200):
    
    UM  = domain_adaptation_task
    cc  = repetition
    SpC = sample_per_class
    
    print ('Creating pairs for repetition: '+str(cc)+' and sample_per_class: '+str(sample_per_class))
    Training_P=[]
    Training_N=[]

    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if y_train_source[trs]==y_train_target[trt]:
                Training_P.append([trs,trt])
            else:
                Training_N.append([trs,trt])

    random.shuffle(Training_N)
    Training = Training_P+Training_N[:3*len(Training_P)]
    random.shuffle(Training)

    X1=np.zeros([len(Training),n_features],dtype='float32')
    X2=np.zeros([len(Training),n_features],dtype='float32')

    y1=np.zeros([len(Training)])
    y2=np.zeros([len(Training)])
    yc=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:]=X_train_source[in1,:]
        X2[i,:]=X_train_target[in2,:]
        y1[i]=y_train_source[in1]
        y2[i]=y_train_target[in2]
        if y_train_source[in1]==y_train_target[in2]:
            yc[i]=1

    # if not os.path.exists(CCSA_path):
    #     os.makedirs(CCSA_path)

    # np.save(CCSA_path+'/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X1)
    # np.save(CCSA_path+'/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X2)

    # np.save(CCSA_path+'/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y1)
    # np.save(CCSA_path+'/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y2)
    # np.save(CCSA_path+'/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', yc)
    
    return X1,X2,y1,y2,yc

def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()

# def CSA_training_the_model(model,CCSA_path,domain_adaptation_task,
#                         repetition,sample_per_class, batch_size,
#                         X_val_target, Y_val_target,
#                         X_test, y_test,
#                         X_train_target, y_train_target,
#                         X_train_source, y_train_source):
    
#     nb_classes=2
#     UM = domain_adaptation_task
#     cc = repetition
#     SpC = sample_per_class
#     y_test = torch.tensor(y_test)
#     y_test = torch.one_hot(y_test, num_classes=nb_classes)
#     Y_val_target = torch.tensor(Y_val_target)
#     Y_val_target = torch.one_hot(Y_val_target, num_classes=nb_classes)

#     X1 = np.load(CCSA_path+'/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
#     X2 = np.load(CCSA_path+'/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
#     y1 = np.load(CCSA_path+'/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
#     y2 = np.load(CCSA_path+'/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
#     yc = np.load(CCSA_path+'/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    
#     y1 = torch.tensor(y1)
#     y1 = torch.one_hot(y1, nb_classes)
#     y2 = torch.tensor(y2)
#     y2 = torch.one_hot(y2, nb_classes)
    
#     return X1,X2,y1,y2,yc,y_test,Y_val_target


