# -*- coding: utf-8 -*-
"""
@author: tsharma2
"""

import theano
from keras import Input, Model
from keras.layers import Dropout, Dense, Activation, Lambda
from keras.optimizers import SGD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np

from model.CCSA import Initialization
from model.mlp import get_k_best, MLP

def run_cv(seed, fold, X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat,
           feature_type, val_size=0, pretrain_set=None,
           batch_size=32, k=-1,
           learning_rate=0.01, lr_decay=0.0,
           dropout=0.5, n_epochs=100, momentum=0.9,
           L1_reg=0.001, L2_reg=0.001,
           hiddenLayers=[128,64]):
    
    #X_w = pretrain_set.get_value(borrow=True) if k > 0 and pretrain_set else None
    X_w = pretrain_set.get_value(borrow=True) if pretrain_set else None
    #print('the size of pretrain set is:')
    #print(np.shape(X_w))
    m = X.shape[1] if k < 0 else k
    columns = list(range(m))
    columns.extend(['scr', 'R', 'G', 'Y'])
    df = pd.DataFrame(columns=columns)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X, GRy_strat):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        G_train, G_test = G[train_index], G[test_index]
        strat_train, strat_test = GRy_strat[train_index], GRy_strat[test_index]
        
        if k > 0:
            if len(np.shape(feature_type))==0:
                print('Single omics feature')
                k_best = SelectKBest(f_classif, k=k)
                k_best.fit(X_train, Y_train)
                X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
            else:
                print('Combination of two omics features')
                k_best_1 = SelectKBest(f_classif, k=int(k/2))
                k_best_2 = SelectKBest(f_classif, k=int(k/2))
                feat_1 = feature_type[0]
                if feat_1=='Protein':
                    f_num = 189
                elif feat_1=='mRNA':
                    f_num = 17176
                elif feat_1=='MicroRNA':
                    f_num = 662
                elif feat_1=='Methylation':
                    f_num = 11882
                X_train_feat_1 = X_train[:,0:f_num]
                X_train_feat_2 = X_train[:,f_num:]
                X_test_feat_1 = X_test[:,0:f_num]
                X_test_feat_2 = X_test[:,f_num:]
                k_best_1.fit(X_train_feat_1, Y_train)
                k_best_2.fit(X_train_feat_2, Y_train)
                X_train_feat_1, X_test_feat_1 = k_best_1.transform(X_train_feat_1), k_best_1.transform(X_test_feat_1)
                X_train_feat_2, X_test_feat_2 = k_best_2.transform(X_train_feat_2), k_best_2.transform(X_test_feat_2)
                #print(np.shape(X_train_feat_1),np.shape(X_train_feat_2))
                #print(np.shape(X_test_feat_1),np.shape(X_test_feat_2))
                X_train = np.concatenate((X_train_feat_1,X_train_feat_2),axis=1)
                X_test = np.concatenate((X_test_feat_1,X_test_feat_2),axis=1)
                #print(np.shape(X_train))
                #print(np.shape(X_test))
                
            if pretrain_set:
                if len(np.shape(feature_type))==0:
                    X_base = k_best.transform(X_w)
                else:
                    if feat_1=='Protein':
                        f_num = 189
                    elif feat_1=='mRNA':
                        f_num = 17176
                    elif feat_1=='MicroRNA':
                        f_num = 662
                    elif feat_1=='Methylation':
                        f_num = 11882
                    X_w_feat_1 = X_w[:,0:f_num]
                    X_w_feat_2 = X_w[:,f_num:]
                    X_base_feat_1 = k_best_1.transform(X_w_feat_1)
                    X_base_feat_2 = k_best_2.transform(X_w_feat_2)
                    X_base = np.concatenate((X_base_feat_1,X_base_feat_2),axis=1)
                    #print(np.shape(X_base))
                pretrain_set = theano.shared(np.array(X_base), name='pretrain_set', borrow=True)
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=0, stratify=strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)
        #print('The no. of features are: '+str(np.shape(X_train)[1]))
        
        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                         L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        if pretrain_set:
            pretrain_config = {'pt_batchsize': 32, 'pt_lr': 0.01, 'pt_epochs': 500, 'corruption_level': 0.3}
            classifier.pretrain(pretrain_set=pretrain_set, pretrain_config=pretrain_config)
            classifier.tune(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        else:
            classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        X_scr = classifier.get_score(X_test)
        array1 = np.column_stack((X_test, X_scr[:,1], R_test, G_test, Y_test))
        df_temp1 = pd.DataFrame(array1, index=list(test_index), columns=columns)
        df = df.append(df_temp1)

    return df

def run_mixture_cv(seed, dataset, ethnicgroups,
                   feature_type, fold=3, k=-1,
                   val_size=0, batch_size=32, momentum=0.9,
                   learning_rate=0.01, lr_decay=0.0,
                   dropout=0.5, n_epochs=100, save_to=None,
                   L1_reg=0.001, L2_reg=0.001, 
                   hiddenLayers=[128, 64]):
    
    #X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = data_select(Source, Target, dataset, ethnicgroups, genders)
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat,
                feature_type, val_size=val_size, batch_size=batch_size, k=k, momentum=momentum,
                learning_rate=learning_rate, lr_decay=lr_decay,
                dropout=dropout, n_epochs=n_epochs,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    ## Mixture 0: AUROC for EA+MG
    y_test_0, y_scr_0 = list(df['Y'].values), list(df['scr'].values)
    ## Mixture 1: AUROC for EA
    y_test_1, y_scr_1 = list(df.loc[((df['R']==ethnicgroups[0])),'Y'].values), \
                        list(df.loc[((df['R']==ethnicgroups[0])),'scr'].values)
    ## Mixture 2: AUROC for MG
    y_test_2, y_scr_2 = list(df.loc[((df['R']==ethnicgroups[1])),'Y'].values), \
                        list(df.loc[((df['R']==ethnicgroups[1])),'scr'].values)
    
    A_CI, W_CI, B_CI = \
        roc_auc_score(y_test_0, y_scr_0, average='weighted'), \
        roc_auc_score(y_test_1, y_scr_1, average='weighted'), \
        roc_auc_score(y_test_2, y_scr_2, average='weighted')
    res = {'A_Auc':A_CI,'W_Auc':W_CI,'B_Auc':B_CI}
    df = pd.DataFrame(res, index=[seed])
    #print(res)
    
    return df

def run_one_race_cv(seed, dataset, feature_type, 
                    fold=3,  k=-1, val_size=0, batch_size=32,
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, save_to=None,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    
    df = run_cv(seed, fold, X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat,
                feature_type, val_size=val_size, batch_size=batch_size, k=k,
                learning_rate=learning_rate,
                lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg,
                hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    #print(res)
    
    return df

def run_CCSA_transfer(seed, Source, Target, dataset, groups,
                      CCSA_path, feature_type, n_features,
                      fold=3, alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[128, 64], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      sample_per_class=2, repetition=1,
                      SourcePairs=False):
    
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    
    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y
    df['G'] = G
    df['GRY'] = GRy_strat
    # domain adaptation - source group
    df_train = df[df['R']==groups[0]]
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values
 	#test groups
    df_test = df[df['R']==groups[1]]
    df_b_y = df_test['Y']
    df_b_R = df_test['R'].values.ravel()
    df_b_G = df_test['G'].values.ravel()
    df_test = df_test.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_test = df_b_y.values.ravel()
    X_test = df_test.values
    if n_features > 0 and n_features < X_test.shape[1]:
        #print('n_features is:')
        #print(n_features)
        if len(np.shape(feature_type))==0:
            print('Single omics feature')
            X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
        else:
            print('Combination of two omics features')
            feat_1 = feature_type[0]
            if feat_1=='Protein':
                f_num = 189
            elif feat_1=='mRNA':
                f_num = 17176
            elif feat_1=='MicroRNA':
                f_num = 662
            elif feat_1=='Methylation':
                f_num = 11882
            X_train_source_feat_1 = X_train_source[:,0:f_num]
            X_train_source_feat_2 = X_train_source[:,f_num:]
            X_test_feat_1 = X_test[:,0:f_num]
            X_test_feat_2 = X_test[:,f_num:]
            X_train_source_feat_1, X_test_feat_1 = get_k_best(X_train_source_feat_1, Y_train_source, X_test_feat_1, int(n_features/2))
            X_train_source_feat_2, X_test_feat_2 = get_k_best(X_train_source_feat_2, Y_train_source, X_test_feat_2, int(n_features/2))
            X_train_source = np.concatenate((X_train_source_feat_1,X_train_source_feat_2),axis=1)
            X_test = np.concatenate((X_test_feat_1,X_test_feat_2),axis=1)
            #print(np.shape(X_train_source_feat_1),np.shape(X_train_source_feat_2))
            #print(np.shape(X_test_feat_1),np.shape(X_test_feat_2))
            #print(np.shape(Y_train_source))
            #print(np.shape(X_train_source))
            #print(np.shape(X_test))
    else:
        n_features = X_test.shape[1]
    if sample_per_class==None:
        samples_provided = 'No'
    else: 
        samples_provided = 'Yes'
    df_score = pd.DataFrame(columns=['scr', 'Y', 'R', 'G'])
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]
        R_train_target_full, R_test_target = df_b_R[train_index], df_b_R[test_index]
        G_train_target_full, G_test_target = df_b_G[train_index], df_b_G[test_index]
        if samples_provided=='No':
            maxallowedsamples_un = np.unique(Y_train_target_full,return_counts=True)
            maxallowedsamples = min(maxallowedsamples_un[1])
            #print('Max samples are allowed: '+ str(maxallowedsamples))
            X_train1,X_val1,Y_train1,Y_val1 = train_test_split(X_train_target_full,Y_train_target_full,random_state=None)
            target_samples_count = np.unique(Y_train1,return_counts=True)
            min_target_samples_count = min(target_samples_count[1])
            sample_per_class = min_target_samples_count
            #print('Sample per class is : '+str(sample_per_class))
            if sample_per_class==1:
                sample_per_class = 2
            elif sample_per_class>2:
                if sample_per_class>maxallowedsamples:
                    sample_per_class = maxallowedsamples
        #print('==================================')
        #print('Sample per class is : '+str(sample_per_class))
        #print('==================================')
        index0 = np.where(Y_train_target_full == 0)
        index1 = np.where(Y_train_target_full == 1)
        target_samples = []
        target_samples.extend(index0[0][0:sample_per_class])
        target_samples.extend(index1[0][0:sample_per_class])
        X_train_target = X_train_target_full[target_samples]
        Y_train_target = Y_train_target_full[target_samples]
        X_val_target = [e for idx, e in enumerate(X_train_target_full) if idx not in target_samples]
        Y_val_target = [e for idx, e in enumerate(Y_train_target_full) if idx not in target_samples]
        best_score, best_Auc = train_and_predict(groups, 
                                     X_train_target, Y_train_target,
                                     X_train_source, Y_train_source,
                                     X_val_target, Y_val_target,
                                     X_test_target, Y_test_target,
                                     CCSA_path,
                                     sample_per_class=sample_per_class,
                                     alpha=alpha, learning_rate=learning_rate,
                                     hiddenLayers=hiddenLayers, dr=dr,
                                     momentum=momentum, decay=decay,
                                     batch_size=batch_size,
                                     repetition=repetition,
                                     n_features=n_features,
                                     SourcePairs=SourcePairs)
        array = np.column_stack((best_score, Y_test_target, R_test_target, G_test_target))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y', 'R', 'G'])
        df_score = df_score.append(df_temp)
    
    auc = roc_auc_score(list(df_score['Y'].values), list(df_score['scr'].values))
    res = {'TL_Auc': auc}
    df = pd.DataFrame(res, index=[seed])
    #print(res)
    
    return df


# let's run the experiments when 1 target sample per calss is available in training.
# you can run the experiments for sample_per_class=1, ... , 7.
# sample_per_class = 2
# Running the experiments for repetition 5. In the paper we reported the average acuracy.
# We run the experiments for repetition=0,...,9 and take the average
# repetition = 2
def train_and_predict(groups,
                      X_train_target, y_train_target,
                      X_train_source, y_train_source,
                      X_val_target, Y_val_target,
                      X_test, y_test, CCSA_path,
                      repetition, sample_per_class,
                      alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      n_features = 400,
                      SourcePairs=False):
    
    # size of input variable for each patient
    domain_adaptation_task = groups[0]+'_to_'+groups[1]
    input_shape = (n_features,)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 2
    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = alpha

    # Having two streams. One for source and one for target.
    model1 = Initialization.Create_Model(hiddenLayers=hiddenLayers, dr=dr)
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    processed_a = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(processed_a)
    out1 = Activation('softmax', name='classification')(out1)

    distance = Lambda(Initialization.euclidean_distance,
                      output_shape=Initialization.eucl_dist_output_shape,
                      name='CSA')([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay) # momentum=0., decay=0.
    model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer=optimizer,
                  loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    #print('Domain Adaptation Task: ' + domain_adaptation_task)
    # for repetition in range(10):
    Initialization.Create_Pairs(CCSA_path, domain_adaptation_task,
                                repetition, sample_per_class,
                                X_train_target, y_train_target,
                                X_train_source, y_train_source,
                                n_features=n_features,
                                SourcePairs=SourcePairs)
    best_score, best_Auc = Initialization.training_the_model(model, CCSA_path,
                                                             domain_adaptation_task,
                                                             repetition,
                                                             sample_per_class,batch_size,
                                                             X_val_target,
                                                             Y_val_target,
                                                             X_test, y_test,
                                                             SourcePairs=SourcePairs)
    #print('Best AUC for {} target sample per class and repetition {} is {}.'.format(sample_per_class, repetition, best_Auc))
    
    return best_score, best_Auc

def run_unsupervised_transfer_cv(seed, Source, Target, dataset, groups,
                                 feature_type, fold=3,
                                 val_size=0, k=-1,
                                 batch_size=32, save_to=None,
                                 learning_rate=0.01, lr_decay=0.0,
                                 dropout=0.5, n_epochs=100,
                                 L1_reg=0.001, L2_reg=0.001,
                                 hiddenLayers=[128, 64]):
    
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    
    idx = (R==groups[0])
    X_s, Y_s = X[idx==True], Y[idx==True]
    pretrain_set = (X_s, Y_s)
    idx = (R==groups[1])
    X_t, Y_t, R_t, y_sub_t, y_strat_t, G_t, Gy_strat_t, GRy_strat_t = X[idx==True], Y[idx==True], R[idx==True], y_sub[idx==True], y_strat[idx==True], G[idx==True], Gy_strat[idx==True], GRy_strat[idx==True]
    
    pretrain_set = theano.shared(X_s, name='pretrain_set', borrow=True)
    #print('pretrain_set in unsupervised loop is:')
    #print(pretrain_set)
    
    df = run_cv(seed, fold, X_t, Y_t, R_t, y_sub_t, y_strat_t, G_t, Gy_strat_t, GRy_strat_t,
                feature_type, pretrain_set=pretrain_set,
                val_size=val_size, batch_size=batch_size, k=k, n_epochs=n_epochs,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    
    if save_to:
        df.to_csv(save_to)
    
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'TL_Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    
    return df

def run_supervised_transfer_cv(seed, Source, Target, dataset,
                               groups, feature_type, fold=3,
                               val_size=0, k=-1,
                               batch_size=32,
                               learning_rate=0.01, lr_decay=0.0,
                               dropout=0.5, tune_epoch=200,
                               tune_lr=0.002, train_epoch=1000,
                               L1_reg=0.001, L2_reg=0.001,
                               hiddenLayers=[128, 64], tune_batch=10,
                               momentum=0.9):
    
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    
    idx = (R==groups[0])
    X_s, Y_s = X[idx==True], Y[idx==True]
    pretrain_set = (X_s, Y_s)
    idx = (R==groups[1])
    X_t, Y_t, R_t, G_t, GRy_strat_t = X[idx==True], Y[idx==True], R[idx==True], G[idx==True], GRy_strat[idx==True]
    
    df = pd.DataFrame(columns=['scr', 'R', 'G', 'Y'])
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X_t, GRy_strat_t):
        X_train, X_test = X_t[train_index], X_t[test_index]
        Y_train, Y_test = Y_t[train_index], Y_t[test_index]
        R_train, R_test = R_t[train_index], R_t[test_index]
        G_train, G_test = G_t[train_index], G_t[test_index]
        strat_train, strat_test = GRy_strat_t[train_index], GRy_strat_t[test_index]
        if k > 0:
            if len(np.shape(feature_type))==0:
                print('Single omics feature')
                k_best = SelectKBest(f_classif, k=k)
                k_best.fit(X_train, Y_train)
                X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
                X_base = k_best.transform(X_s)
            else:
                print('Combination of two omics features')
                k_best_1 = SelectKBest(f_classif, k=int(k/2))
                k_best_2 = SelectKBest(f_classif, k=int(k/2))
                feat_1 = feature_type[0]
                if feat_1=='Protein':
                    f_num = 189
                elif feat_1=='mRNA':
                    f_num = 17176
                elif feat_1=='MicroRNA':
                    f_num = 662
                elif feat_1=='Methylation':
                    f_num = 11882
                X_train_feat_1 = X_train[:,0:f_num]
                X_train_feat_2 = X_train[:,f_num:]
                X_test_feat_1 = X_test[:,0:f_num]
                X_test_feat_2 = X_test[:,f_num:]
                X_s_feat_1 = X_s[:,0:f_num]
                X_s_feat_2 = X_s[:,f_num:]
                k_best_1.fit(X_train_feat_1, Y_train)
                k_best_2.fit(X_train_feat_2, Y_train)
                X_train_feat_1, X_test_feat_1 = k_best_1.transform(X_train_feat_1), k_best_1.transform(X_test_feat_1)
                X_train_feat_2, X_test_feat_2 = k_best_2.transform(X_train_feat_2), k_best_2.transform(X_test_feat_2)
                X_base_feat_1 = k_best_1.transform(X_s_feat_1)
                X_base_feat_2 = k_best_2.transform(X_s_feat_2)
                #print(np.shape(X_train_feat_1),np.shape(X_train_feat_2))
                #print(np.shape(X_test_feat_1),np.shape(X_test_feat_2))
                #print(np.shape(X_base_feat_1),np.shape(X_base_feat_2))
                X_train = np.concatenate((X_train_feat_1,X_train_feat_2),axis=1)
                X_test = np.concatenate((X_test_feat_1,X_test_feat_2),axis=1)
                X_base = np.concatenate((X_base_feat_1,X_base_feat_2),axis=1)
                #print(np.shape(X_train))
                #print(np.shape(X_test))
                #print(np.shape(X_base))
            pretrain_set = (X_base, Y_s)
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val, R_train, R_val, G_train, G_val, \
                        strat_train, strat_val = train_test_split(X_train, Y_train, R_train, G_train, strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)
        n_in = X_train.shape[1]
        #print('The no. of features are: '+str(np.shape(X_train)[1]))
        classifier = MLP(n_in=n_in, learning_rate=learning_rate,
                lr_decay=lr_decay, dropout=dropout, L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers)
        classifier.train(pretrain_set, n_epochs=train_epoch, batch_size=batch_size)
        classifier.learning_rate = tune_lr
        classifier.tune(train_data, valid_data=valid_data, batch_size=tune_batch, n_epochs=tune_epoch)
        scr = classifier.get_score(X_test)
        array = np.column_stack((scr[:, 1], R_test, G_test, Y_test))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'R', 'G', 'Y'])
        df = df.append(df_temp)

    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'TL_Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
    
    return df
    
def run_naive_transfer_cv(seed, Source, Target, dataset, ethnicgroups,
                        feature_type, fold=3, k=-1, val_size=0,
                        batch_size=32, momentum=0.9,
                        learning_rate=0.01, lr_decay=0.0, 
                        dropout=0.5, n_epochs=100,
                        save_to=None, L1_reg=0.001,
                        L2_reg=0.001, hiddenLayers=[128, 64]):
    
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    
    m = X.shape[1] if k < 0 else k
    columns = list(range(m))
    columns.extend(['scr', 'R', 'G', 'Y'])
    df = pd.DataFrame(columns=columns)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    
    for train_index, test_index in kf.split(X, GRy_strat):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        G_train, G_test = G[train_index], G[test_index]
        strat_train, strat_test = GRy_strat[train_index], GRy_strat[test_index]
        if k > 0:
            if len(np.shape(feature_type))==0:
                print('Single omics feature')
                k_best = SelectKBest(f_classif, k=k)
                k_best.fit(X_train, Y_train)
                X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
            else:
                print('Combination of two omics features')
                k_best_1 = SelectKBest(f_classif, k=int(k/2))
                k_best_2 = SelectKBest(f_classif, k=int(k/2))
                feat_1 = feature_type[0]
                if feat_1=='Protein':
                    f_num = 189
                elif feat_1=='mRNA':
                    f_num = 17176
                elif feat_1=='MicroRNA':
                    f_num = 662
                elif feat_1=='Methylation':
                    f_num = 11882
                X_train_feat_1 = X_train[:,0:f_num]
                X_train_feat_2 = X_train[:,f_num:]
                X_test_feat_1 = X_test[:,0:f_num]
                X_test_feat_2 = X_test[:,f_num:]
                k_best_1.fit(X_train_feat_1, Y_train)
                k_best_2.fit(X_train_feat_2, Y_train)
                X_train_feat_1, X_test_feat_1 = k_best_1.transform(X_train_feat_1), k_best_1.transform(X_test_feat_1)
                X_train_feat_2, X_test_feat_2 = k_best_2.transform(X_train_feat_2), k_best_2.transform(X_test_feat_2)
                #print(np.shape(X_train_feat_1),np.shape(X_train_feat_2))
                #print(np.shape(X_test_feat_1),np.shape(X_test_feat_2))
                X_train = np.concatenate((X_train_feat_1,X_train_feat_2),axis=1)
                X_test = np.concatenate((X_test_feat_1,X_test_feat_2),axis=1)
                #print(np.shape(X_train))
                #print(np.shape(X_test))
        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val, R_train, R_val, G_train, G_val, \
                        strat_train, strat_val = train_test_split(X_train, Y_train, R_train, G_train, strat_train)
            valid_data = (X_val, Y_val)
            idx = (R_val==ethnicgroups[0])
            X_val, Y_val = X_val[idx==True], Y_val[idx==True]
            valid_data = (X_val, Y_val)
        
        idx = (R_train==ethnicgroups[0])
        X_train, Y_train = X_train[idx==True], Y_train[idx==True] 
        train_data = (X_train, Y_train)
        n_in = X_train.shape[1]
        #print('The no. of features are: '+str(np.shape(X_train)[1]))
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                         L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        idx = (R_test==ethnicgroups[1])
        X_test, Y_test, R_test, G_test, strat_test = X_test[idx], Y_test[idx], R_test[idx], G_test[idx], strat_test[idx]
        X_scr = classifier.get_score(X_test)
        array1 = np.column_stack((X_test, X_scr[:,1], R_test, G_test, Y_test))
        df_temp1 = pd.DataFrame(array1, columns=columns)
        df = df.append(df_temp1)

    if save_to:
        df.to_csv(save_to)
    
    y_test_b, y_scr_b = list(df.loc[df['R']==ethnicgroups[1], 'Y'].values), list(df.loc[df['R']==ethnicgroups[1], 'scr'].values)
    B_CI = roc_auc_score(y_test_b, y_scr_b, average='weighted')
    res = {'NT_Auc': B_CI}
    df = pd.DataFrame(res, index=[seed])
    
    return df

import torch
import dataloader
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from model import fada_model
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
## FADA (torch):
def FADA_classification(seed, Source, Target, dataset, ethnicgroups, 
                      checkpt_path, feature_type, n_features,
                      fold=3, alpha=0.3, learning_rate = 0.01,
                      hiddenLayers=[128, 64], dr=0.5,
                      momentum=0.9, decay=0.0, batch_size=20,
                      sample_per_class=2, EarlyStop=False, DCD_optimizer='SGD',
                      patience=100, n_epochs=100,
                      L1_reg=0.001, L2_reg=0.001):
        
    valid_data = True if EarlyStop else None
    X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
    m = X.shape[1] if n_features < 0 else n_features
    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y
    df['G'] = G
    df['GRY'] = GRy_strat
    # domain adaptation - source group
    df_train = df[df['R']==ethnicgroups[0]]
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values
 	#test groups
    df_test = df[df['R']==ethnicgroups[1]]
    df_b_y = df_test['Y']
    df_b_R = df_test['R'].values.ravel()
    df_b_G = df_test['G'].values.ravel()
    df_test = df_test.drop(columns=['Y', 'R', 'G', 'GRY'])
    Y_test = df_b_y.values.ravel()
    X_test = df_test.values
    
    if n_features > 0 and n_features < X_test.shape[1]:
        #print('n_features is:')
        #print(n_features)
        if len(np.shape(feature_type))==0:
            print('Single omics feature')
            X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
        else:
            print('Combination of two omics features')
            feat_1 = feature_type[0]
            if feat_1=='Protein':
                f_num = 189
            elif feat_1=='mRNA':
                f_num = 17176
            elif feat_1=='MicroRNA':
                f_num = 662
            elif feat_1=='Methylation':
                f_num = 11882
            X_train_source_feat_1 = X_train_source[:,0:f_num]
            X_train_source_feat_2 = X_train_source[:,f_num:]
            X_test_feat_1 = X_test[:,0:f_num]
            X_test_feat_2 = X_test[:,f_num:]
            X_train_source_feat_1, X_test_feat_1 = get_k_best(X_train_source_feat_1, Y_train_source, X_test_feat_1, int(n_features/2))
            X_train_source_feat_2, X_test_feat_2 = get_k_best(X_train_source_feat_2, Y_train_source, X_test_feat_2, int(n_features/2))
            X_train_source = np.concatenate((X_train_source_feat_1,X_train_source_feat_2),axis=1)
            X_test = np.concatenate((X_test_feat_1,X_test_feat_2),axis=1)
            #print(np.shape(X_train_source_feat_1),np.shape(X_train_source_feat_2))
            #print(np.shape(X_test_feat_1),np.shape(X_test_feat_2))
            #print(np.shape(Y_train_source))
            #print(np.shape(X_train_source))
            #print(np.shape(X_test))
    else:
        n_features = X_test.shape[1]
    
    if sample_per_class==None:
        samples_provided = 'No'
    else: 
        samples_provided = 'Yes'
        
    trainData = dataloader.CSVDataset(X_train_source,Y_train_source)
    train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    X_s = torch.tensor(X_train_source)
    Y_s = torch.tensor(Y_train_source,dtype=torch.int64)
    df_score = pd.DataFrame(columns=['scr', 'Y', 'R', 'G'])
    kf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]
        R_train_target_full, R_test_target = df_b_R[train_index], df_b_R[test_index]
        G_train_target_full, G_test_target = df_b_G[train_index], df_b_G[test_index]
        if samples_provided=='No':
            maxallowedsamples_un = np.unique(Y_train_target_full,return_counts=True)
            maxallowedsamples = min(maxallowedsamples_un[1])
            #print('Max samples are allowed: '+ str(maxallowedsamples))
            X_train1,X_val1,Y_train1,Y_val1 = train_test_split(X_train_target_full,Y_train_target_full,random_state=None)
            target_samples_count = np.unique(Y_train1,return_counts=True)
            min_target_samples_count = min(target_samples_count[1])
            sample_per_class = min_target_samples_count
            #print('Sample per class is : '+str(sample_per_class))
            if sample_per_class==1:
                sample_per_class = 2
            elif sample_per_class>2:
                if sample_per_class>maxallowedsamples:
                    sample_per_class = maxallowedsamples
        #print('==================================')
        #print('Sample per class is : '+str(sample_per_class))
        #print('==================================')
        
        train_targetData = dataloader.CSVDataset(X_train_target_full,Y_train_target_full)
        X_t,Y_t,X_val,Y_val,valid_data = dataloader.create_target_samples_cancer(train_targetData,sample_per_class)
        
        if valid_data==True:
            #Y_val = F.one_hot(Y_val.to(torch.int64), num_classes=nb_classes)
            val_dataloader = dataloader.CSVDataset(X_val,Y_val)
            val_dataloader = DataLoader(val_dataloader,batch_size=len(X_val))
                
        net = fada_model.Network(in_features_data=m,nb_classes=2,dropout=dr,hiddenLayers=[128,64])
        net.to(device)
        loss_fn = torch.nn.BCELoss()
        
        discriminator = fada_model.DCD(h_features=64,input_features=128)#128=64*2 i.e. twice the output of classifier --> stacking
        discriminator.to(device)
        loss_discriminator = torch.nn.CrossEntropyLoss()
        
        #STEP 1:
        if DCD_optimizer=='SGD':
            optimizer1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer1 = torch.optim.Adam(list(net.parameters()),lr=learning_rate)
        #STEP 2:
        if DCD_optimizer=='SGD':
            optimizer_D = torch.optim.SGD(discriminator.parameters(),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        #STEP 3:
        if DCD_optimizer=='SGD':
            optimizer_g_h1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_g_h1 = torch.optim.Adam(list(net.parameters()),lr=learning_rate)
        if DCD_optimizer=='SGD':
            optimizer_d = torch.optim.SGD(discriminator.parameters(),lr=learning_rate,momentum=momentum)
        elif DCD_optimizer=='Adam':
            optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        
        # optimizer1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        # optimizer_g_h1 = torch.optim.SGD(list(net.parameters()),lr=learning_rate,momentum=momentum)
        optimizer_d = torch.optim.Adam(discriminator.parameters(),lr=learning_rate)
        
        ###################
        # STEP 1:
        ###################
        train_losses_s1 = []
        avg_train_losses_s1 = []
        valid_losses_s1 = []
        valid_loss_s1 = []
        avg_valid_losses_s1 = []
        
        # initialize the early_stopping object
        early_stopping1 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            for data,labels in train_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.long)
                #labels = F.one_hot(labels.to(torch.int64), num_classes=nb_classes)
                optimizer1.zero_grad()
                y_pred,_ = net(data)
                loss = loss_fn(y_pred[:,1],labels.float())
                l1_norm = sum(p.abs().sum() for p in net.parameters())
                l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                loss.backward()
                optimizer1.step()
                train_losses_s1.append(loss.item())
                
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses_s1.append(v_loss.numpy())
            
            # print training/validation statistics
            train_loss_s1 = np.average(train_losses_s1)
            avg_train_losses_s1.append(train_loss_s1)
            if valid_data==True:
                valid_loss_s1 = np.average(valid_losses_s1)
                avg_valid_losses_s1.append(valid_loss_s1)
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss_s1: {train_loss_s1:.5f} ')
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_s1 = []
            valid_losses_s1 = []
            
            if valid_data==True:
                if EarlyStop==True:
                    early_stopping1(valid_loss_s1, net)
                    if early_stopping1.early_stop:
                        print("Early stopping")
                        break
                    
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                net.load_state_dict(torch.load(checkpt_path))
        
        ###################
        # STEP 2:
        ###################
        train_losses_s2 = []
        avg_train_losses_s2 = []
        valid_losses_s2 = []
        valid_loss_s2 = []
        avg_valid_losses_s2 = []
        
        # initialize the early_stopping object
        early_stopping2 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            groups,aa = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=n_epochs+epoch)
            n_iters = 4 * len(groups[1])
            index_list = torch.randperm(n_iters)
            mini_batch_size = n_iters//2
            X1 = [];
            X2 = [];
            ground_truths = []
            for index in range(n_iters):
                #ground_truth=index_list[index]//len(groups[1])
                ground_truth = torch.div(index_list[index],len(groups[1]),rounding_mode='floor')
                x1,x2 = groups[ground_truth][index_list[index]-len(groups[1])*ground_truth]
                X1.append(x1)
                X2.append(x2)
                ground_truths.append(ground_truth)
        
                #select data for a mini-batch to train
                if (index+1)%mini_batch_size==0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths = torch.LongTensor(ground_truths)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    ground_truths = F.one_hot(ground_truths.to(torch.int64), num_classes=4)#4 groups
                    ground_truths = ground_truths.to(device)
                    optimizer_D.zero_grad()
                    _,feature_out1 = net(X1)
                    _,feature_out2 = net(X2)
                    X_cat = torch.cat([feature_out1,feature_out2],1)
                    y_pred = discriminator(X_cat.detach())
                    loss = loss_discriminator(y_pred,ground_truths.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                    loss.backward()
                    optimizer_D.step()
                    train_losses_s2.append(loss.item())
                    
                    X1 = []
                    X2 = []
                    ground_truths = []
            
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses_s2.append(v_loss.numpy())
            
            # print training/validation statistics
            train_loss_s2 = np.average(train_losses_s2)
            avg_train_losses_s2.append(train_loss_s2)
            if valid_data==True:
                valid_loss_s2 = np.average(valid_losses_s2)
                avg_valid_losses_s2.append(valid_loss_s2)
            
            epoch_len = len(str(n_epochs))
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + f'train_loss_s2: {train_loss_s2:.5f} ')
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_s2 = []
            valid_losses_s2 = []
            
            if valid_data==True:
                if EarlyStop==True:
                    early_stopping2(valid_loss_s2, discriminator)
                    if early_stopping2.early_stop:
                        print("Early stopping")
                        break
        
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                discriminator.load_state_dict(torch.load(checkpt_path))
        
        ###################
        # STEP 3:
        ###################
        train_losses_gh = []
        train_losses_dcd = []
        valid_losses = []
        valid_loss = []
        avg_train_losses_gh = []
        avg_train_losses_dcd = []
        avg_valid_losses = [] 
        
        # initialize the early_stopping object
        early_stopping3 = EarlyStopping(patience=patience,verbose=True,path=checkpt_path)
        for epoch in range(n_epochs):
            #---training g and h , DCD is frozen        
            groups, groups_y = dataloader.sample_groups(X_s,Y_s,X_t,Y_t,seed=n_epochs+epoch)
            G1, G2, G3, G4 = groups
            Y1, Y2, Y3, Y4 = groups_y
            groups_2 = [G2, G4]
            groups_y_2 = [Y2, Y4]
            n_iters = 2 * len(G2)
            index_list = torch.randperm(n_iters)
            n_iters_dcd = 4 * len(G2)
            index_list_dcd = torch.randperm(n_iters_dcd)
            mini_batch_size_g_h = n_iters//2
            mini_batch_size_dcd = n_iters_dcd//2
            X1 = []
            X2 = []
            ground_truths_y1 = []
            ground_truths_y2 = []
            dcd_labels = []
            for index in range(n_iters):
                #ground_truth=index_list[index]//len(G2)
                ground_truth = torch.div(index_list[index],len(G2),rounding_mode='floor')
                x1, x2 = groups_2[ground_truth][index_list[index] - len(G2) * ground_truth]
                y1, y2 = groups_y_2[ground_truth][index_list[index] - len(G2) * ground_truth]
                dcd_label = 0 if ground_truth==0 else 2
                X1.append(x1)
                X2.append(x2)
                ground_truths_y1.append(y1)
                ground_truths_y2.append(y2)
                dcd_labels.append(dcd_label)
                if (index+1)%mini_batch_size_g_h==0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths_y1 = torch.LongTensor(ground_truths_y1)
                    ground_truths_y2 = torch.LongTensor(ground_truths_y2)
                    dcd_labels = torch.LongTensor(dcd_labels)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    #ground_truths_y1 = F.one_hot(ground_truths_y1.to(torch.int64), num_classes=2)
                    #ground_truths_y2 = F.one_hot(ground_truths_y2.to(torch.int64), num_classes=2)
                    ground_truths_y1 = ground_truths_y1.to(device)
                    ground_truths_y2 = ground_truths_y2.to(device)
                    dcd_labels = F.one_hot(dcd_labels.to(torch.int64), num_classes=4)
                    dcd_labels = dcd_labels.to(device)
                    optimizer_g_h1.zero_grad()
                    y_pred_X1,encoder_X1 = net(X1)
                    y_pred_X2,encoder_X2 = net(X2)
                    X_cat = torch.cat([encoder_X1,encoder_X2],1)
                    y_pred_dcd = discriminator(X_cat)
                    loss_X1 = loss_fn(y_pred_X1[:,1],ground_truths_y1.float())
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    loss_X1 = loss_X1 + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_X2 = loss_fn(y_pred_X2[:,1],ground_truths_y2.float())
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    loss_X2 = loss_X2 + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_dcd = loss_discriminator(y_pred_dcd,dcd_labels.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss_dcd = loss_dcd + L2_reg * l2_norm + L1_reg * l1_norm
                    loss_sum = (loss_X1 + loss_X2 + alpha*loss_dcd)/3
                    # l1_norm = sum(p.abs().sum() for p in net.parameters())
                    # l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    # loss_sum = loss_sum + L2_reg * l2_norm + L1_reg * l1_norm
                    
                    loss_sum.backward()
                    optimizer_g_h1.step()
                    train_losses_gh.append(loss_sum.item())
        
                    X1 = []
                    X2 = []
                    ground_truths_y1 = []
                    ground_truths_y2 = []
                    dcd_labels = []
            
            #----training dcd ,g and h frozen
            X1 = []
            X2 = []
            ground_truths = []
            for index in range(n_iters_dcd):
                #ground_truth=index_list_dcd[index]//len(groups[1])
                ground_truth = torch.div(index_list_dcd[index],len(groups[1]),rounding_mode='floor')
                x1, x2 = groups[ground_truth][index_list_dcd[index] - len(groups[1]) * ground_truth]
                X1.append(x1)
                X2.append(x2)
                ground_truths.append(ground_truth)
        
                if (index + 1) % mini_batch_size_dcd == 0:
                    X1 = torch.stack(X1)
                    X2 = torch.stack(X2)
                    ground_truths = torch.LongTensor(ground_truths)
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    ground_truths = F.one_hot(ground_truths.to(torch.int64), num_classes=4)#4 groups
                    ground_truths = ground_truths.to(device)
                    optimizer_d.zero_grad()
                    _,feature_out11 = net(X1)
                    _,feature_out12 = net(X2)
                    X_cat = torch.cat([feature_out11, feature_out12], 1)
                    y_pred = discriminator(X_cat.detach())
                    loss = loss_discriminator(y_pred, ground_truths.float())
                    l1_norm = sum(p.abs().sum() for p in discriminator.parameters())
                    l2_norm = sum(p.pow(2.0).sum() for p in discriminator.parameters())
                    loss = loss + L2_reg * l2_norm + L1_reg * l1_norm
                    loss.backward()
                    optimizer_d.step()
                    train_losses_dcd.append(loss.item())
                    
                    X1 = []
                    X2 = []
                    ground_truths = []
            
            ######################    
            # validate the model #
            ######################
            if valid_data==True:
                with torch.no_grad():
                    for val_data,val_targets in val_dataloader:
                        val_data = val_data.to(device)
                        val_targets = val_targets.to(device)
                        val_targets = val_targets.to(torch.long)
                        val_pred,_ = net(val_data)
                        v_loss = loss_fn(val_pred[:,1],val_targets.float())
                        valid_losses.append(v_loss.numpy())
            
            # print training/validation statistics 
            # calculate average loss over an epoch
            avg_train_losses_gh.append(np.average(train_losses_gh))
            avg_train_losses_dcd.append(np.average(train_losses_dcd))
            if valid_data==True:
                valid_loss = np.average(valid_losses)
                avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(n_epochs))
            train_loss_gh = np.average(train_losses_gh)
            train_loss_dcd = np.average(train_losses_dcd)
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                          f'train_losses_gh: {train_loss_gh:.5f} ' +
                          f'train_losses_dcd: {train_loss_dcd:.5f} ')
            
            #print(print_msg)
            
            # clear lists to track next epoch
            train_losses_gh = []
            train_losses_dcd = []
            valid_losses = []
            
            if valid_data==True:
                if EarlyStop==True:
                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping3(valid_loss, net)
                    if early_stopping3.early_stop:
                        print("Early stopping")
                        break
        
        if valid_data==True:
            if EarlyStop==True:
                # load the last checkpoint with the best model
                net.load_state_dict(torch.load(checkpt_path))
        
        ######################    
        # testing the model #
        ######################
        X_test_target = torch.tensor(X_test_target)
        X_test_target = X_test_target.to(device)
        Y_test_target = torch.tensor(Y_test_target)
        #Y_test_target = F.one_hot(Y_test_target.to(torch.int64), num_classes=nb_classes)
        Y_test_target = Y_test_target.to(device)
        
        with torch.no_grad():
            y_test_pred,_ = net(X_test_target)
            #_, idx = y_test_pred.max(dim=1)
            
        best_score = y_test_pred[:,1]
        array = np.column_stack((best_score, Y_test[test_index], R_test_target, G_test_target))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y', 'R', 'G'])
        df_score = df_score.append(df_temp)
    
    auc = roc_auc_score(list(df_score['Y'].values),
                        list(df_score['scr'].values),
                        average='weighted')
    res = {'TL_DCD_Auc': auc}
    df_DCD_TL = pd.DataFrame(res, index=[seed])
    # print(res)
    
    return df_DCD_TL
    