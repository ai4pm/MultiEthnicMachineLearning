# -*- coding: utf-8 -*-
"""
@author: tsharma2
"""

import argparse
from numpy.random import seed
import pandas as pd
import random as rn
import os
import numpy as np
import time
from preProcess import get_dataset, get_MicroRNA, get_Methylation, standarize_dataset, normalize_dataset, get_n_years, get_independent_data_single, run_cv_gender_race_comb
from classify_all import run_mixture_cv, run_one_race_cv, run_unsupervised_transfer_cv, run_CCSA_transfer, run_supervised_transfer_cv, run_naive_transfer_cv, FADA_classification 
from tensorflow import set_random_seed

seed(11111)
set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

folderISAAC = './'

def main():
    
    feature_type = ("mRNA","Methylation")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cancer_type", type=str, help="Cancer Type")
    parser.add_argument("target", type=str, help="Clinical Outcome Endpoint")
    parser.add_argument("years", type=int, help="Event Time Threhold (Years)")
    parser.add_argument("target_domain", type=str, help="Target Group")
    parser.add_argument("features_count", type=int, help="No. of Features")
    args = parser.parse_args()
    #print(args)
    cancer_type = args.cancer_type
    target = args.target
    years = args.years
    features_count = args.features_count # no. of features to be selected
    source_domain = 'WHITE'
    target_domain = args.target_domain
    groups = (source_domain,target_domain)
    TaskName = 'TCGA-'+cancer_type+'-'+str(feature_type)+'-'+ groups[0]+'-'+groups[1]+'-'+target+'-'+str(years)+'YR'
    out_file_name = folderISAAC + 'Result/' + TaskName + '.xlsx'
    CCSA_path = folderISAAC +'CCSA_data/' + TaskName + '/CCSA_pairs'
    checkpt_path = folderISAAC+'ckpt/FADA_'+TaskName+'_checkpoint.pt'
    if os.path.exists(out_file_name)!=True:
        if len(np.shape(feature_type))==0:
            print('Single omics feature')
        else:
            print('Combination of two omics features')
        # fetch dataset
        if feature_type=='mRNA':
            dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups)
        elif feature_type=='MicroRNA':
            dataset = get_MicroRNA(cancer_type=cancer_type,target=target,groups=groups)
        elif feature_type=='Protein':
            dataset = get_dataset(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups)
        elif feature_type=='Methylation':
            dataset = get_Methylation(cancer_type=cancer_type,target=target,groups=groups)
        else:
            dataset = run_cv_gender_race_comb(cancer_type=cancer_type,feature_type=feature_type,target=target,groups=groups)
        if features_count<=(np.shape(dataset['X'])[1]):
            k = features_count
        else:
            k = -1
        dataset = standarize_dataset(dataset)
        
        print (cancer_type, feature_type, target, years, source_domain, target_domain)
        print('The value of k is: '+str(k))
        print(np.shape(dataset['X']))
        
        dataset = standarize_dataset(dataset)
        data_w = get_independent_data_single(dataset, 'WHITE', groups)
        data_w = get_n_years(data_w, years)
        data_b = get_independent_data_single(dataset, 'MG', groups)
        data_b = get_n_years(data_b, years)
        # dataset_tl_ccsa = normalize_dataset(dataset)
        # dataset_tl_ccsa = get_n_years(dataset_tl_ccsa, years)
        dataset = get_n_years(dataset, years)
        
        X, Y, R, y_sub, y_strat, G, Gy_strat, GRy_strat = dataset
        df = pd.DataFrame(y_strat, columns=['RY'])
        df['R'] = R
        df['Y'] = Y
        print(X.shape)
        print(df['RY'].value_counts())#race with prognosis counts
        print(df['R'].value_counts())#race counts
        print(df['Y'].value_counts())#progonsis counts
        
        ###############################
        # parameters #
        ###############################
        parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9, 'learning_rate':0.01,
                        'lr_decay':0.03, 'dropout':0.5, 'L1_reg': 0.001,'L2_reg': 0.001, 'hiddenLayers': [128,64]}
        parameters_MAJ = {'fold':3, 'k':k, 'batch_size':20, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_MIN = {'fold':3, 'k':k, 'batch_size':4, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.01,
                        'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_NT  = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL1 = {'fold':3, 'k':k, 'batch_size':20, 'momentum':0.9, 'lr_decay':0.03, 'val_size':0.0,
                        'learning_rate':0.01, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64],
                        'train_epoch':100, 'tune_epoch':100, 'tune_lr':0.002, 'tune_batch':10}
        parameters_TL2 = {'fold':3, 'k':k, 'batch_size':10, 'lr_decay':0.03, 'val_size':0.0, 'learning_rate':0.002,
                        'n_epochs':100, 'dropout':0.5, 'L1_reg':0.001, 'L2_reg':0.001, 'hiddenLayers':[128,64]}
        parameters_TL3 = {'fold':3, 'n_features':k, 'alpha':0.3, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[100],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':None, 'SourcePairs':False}
        parameters_TL4 = {'fold':3, 'n_features':k, 'alpha':0.25, 'batch_size':20, 'learning_rate':0.01, 'hiddenLayers':[128,64],
                        'dr':0.5, 'momentum':0.9, 'decay':0.03, 'sample_per_class':None, 'EarlyStop':False,
                        'L1_reg':0.001, 'L2_reg':0.001, 'patience':100, 'n_epochs':100}
        
        res = pd.DataFrame()
        for i in range(20):
            print('###########################')
            print('Interation no.: '+str(i+1))
            print('###########################')
            seed = i
            start_iter = time.time()
            df_mix = run_mixture_cv(seed, dataset, groups, feature_type, **parametrs_mix)
            print('Mixture is done')
            df_w = run_one_race_cv(seed, data_w, feature_type, **parameters_MAJ)
            df_w = df_w.rename(columns={"Auc": "W_ind"})
            print('Independent EA is done.')
            df_b = run_one_race_cv(seed, data_b, feature_type, **parameters_MIN)
            df_b = df_b.rename(columns={"Auc": "B_ind"})
            print('Independent MG is done.')
            df_nt = run_naive_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, feature_type, **parameters_NT)
            print('Naive Transfer is done.')
            df_tl_sup = run_supervised_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, feature_type, **parameters_TL1)
            df_tl_sup = df_tl_sup.rename(columns={"TL_Auc": "TL_sup"})
            print('Supervised is done.')
            df_tl_unsup = run_unsupervised_transfer_cv(seed, 'WHITE', 'MG', dataset, groups, feature_type, **parameters_TL2)
            df_tl_unsup = df_tl_unsup.rename(columns={"TL_Auc": "TL_unsup"})
            print('Unsupervised is done.')
            df_tl_ccsa = run_CCSA_transfer(seed, 'WHITE', 'MG', dataset, groups, CCSA_path, feature_type, **parameters_TL3)
            df_tl_ccsa = df_tl_ccsa.rename(columns={"TL_Auc": "TL_ccsa"})
            print('CCSA is done.')
            df_tl_fada = FADA_classification(seed, 'WHITE', 'MG', dataset, groups, checkpt_path, feature_type, **parameters_TL4)
            df_tl_fada = df_tl_fada.rename(columns={"TL_DCD_Auc":"TL_FADA"})
            print('FADA is done.')
            end_iter = time.time()
            #print("The time of loop execution is :", end_iter-start_iter)
            timeFor_iter = pd.DataFrame({'Time':[end_iter - start_iter]},index=[seed])
            df1 = pd.concat([timeFor_iter,df_mix,df_w, df_b, df_nt, df_tl_sup, df_tl_unsup, df_tl_ccsa, df_tl_fada], sort=False, axis=1)
            res = res.append(df1)
        res.to_excel(out_file_name)

if __name__ == '__main__':
    main()




    
