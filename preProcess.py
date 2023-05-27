# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:53:08 2022

@author: tsharma2
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os.path

path_to_data = 'C:/Users/tsharma2/Dropbox (UTHSC GGI)/Dataset/EssentialData/'

if os.path.exists(path_to_data)==True:
    home_path = path_to_data
else:
    home_path = 'Dataset/EssentialData/'

def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'UCS', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                           'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                           'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                           'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_protein(cancer_type, target, groups):
    
    path = home_path + 'ProteinData/Protein.txt'
    df = pd.read_csv(path, sep='\t', index_col='SampleID')
    df = df.dropna(axis=1)
    tumorTypes = tumor_types(cancer_type)
    df = df[df['TumorType'].isin(tumorTypes)]
    df = df.drop(columns=['TumorType'])
    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new

    return add_race_CT(cancer_type, df, target, groups)

def get_Methylation(cancer_type, target, groups):
    
    MethylationDataPath = home_path + 'MethylationData/Methylation.mat'
    #print(MethylationDataPath)
    MethylationData = loadmat(MethylationDataPath)
    
    # extracting input combinations data...
    X, Y, GeneName, SampleName = MethylationData['X'].astype('float32'), MethylationData['CancerType'], MethylationData['FeatureName'][0], MethylationData['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    MethylationData_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    MethylationData_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    MethylationData_Y = MethylationData_Y[MethylationData_Y['Disease'].isin(tumor_types(cancer_type))]
    MethylationData_in = MethylationData_X.join(MethylationData_Y, how='inner')
    MethylationData_in = MethylationData_in.drop(columns=['Disease'])
    #print('The shape of fetched methylation data initially is:')
    #print(MethylationData_in.shape)
    index = MethylationData_in.index.values
    index_new = [row[:12] for row in index]
    MethylationData_in.index = index_new
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    #print('The shape of fetched methylation data is:')
    #print(MethylationData_in.shape)
    
    # adding race information...
    MethyAncsDataPath = home_path + 'MethylationData/MethylationGenetic.xlsx'
    #print(MethyAncsDataPath)
    # fetching race info from MethylationGenetic.xlsx
    MethyAncsData = [pd.read_excel(MethyAncsDataPath,
                         disease, usecols='A,B',
                         index_col='bcr_patient_barcode',
                         keep_default_na=False)
           for disease in tumor_types(cancer_type)]
    MethyAncsData_race = pd.concat(MethyAncsData)
    race_groups = ['WHITE',
              'BLACK OR AFRICAN AMERICAN',
              'ASIAN',
              'AMERICAN INDIAN OR ALASKA NATIVE',
              'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(race_groups)]
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'WHITE', 'race'] = 'WHITE'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'BLACK OR AFRICAN AMERICAN', 'race'] = 'BLACK'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'ASIAN', 'race'] = 'ASIAN'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'AMERICAN INDIAN OR ALASKA NATIVE', 'race'] = 'NAT_A'
    MethyAncsData_race.loc[MethyAncsData_race['race'] == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'race'] = 'OTHER'
    MethyAncsData_race = MethyAncsData_race[MethyAncsData_race['race'].isin(groups)]
    #print('The shape of race methylation data is:')
    #print(MethyAncsData_race.shape)
    
    # fetching outcome data from MethylationClinInfo.xlsx
    MethyCIDataPath = home_path + 'MethylationData/MethylationClinInfo.xlsx'
    #print(MethyCIDataPath)
    if target=='OS':
        cols = 'A,D,Y,Z'
    elif target == 'DSS':
        cols = 'A,D,AA,AB'
    elif target == 'DFI': # this info is not/very few in methylation data.
        cols = 'A,D,AC,AD'
    elif target == 'PFI':
        cols = 'A,D,AE,AF'
    OutcomeData_M = pd.read_excel(MethyCIDataPath,
                                usecols=cols,dtype={'OS': np.float64},
                                index_col='bcr_patient_barcode')
    
    # adding clinical outcome endpoints data...
    OutcomeData_M.columns = ['G', 'E', 'T']
    OutcomeData_M = OutcomeData_M[OutcomeData_M['E'].isin([0, 1])]
    OutcomeData_M = OutcomeData_M.dropna()
    OutcomeData_M['C'] = 1 - OutcomeData_M['E']
    OutcomeData_M.drop(columns=['E'], inplace=True)
    #print('The shape of outcome methylation data is:')
    #print(OutcomeData_M.shape)
    
    # Keep patients with race information
    MethylationData_in = MethylationData_in.join(MethyAncsData_race, how='inner')
    MethylationData_in = MethylationData_in.dropna(axis='columns')
    #print('The shape of MethylationData_in after race addition is:')
    #print(MethylationData_in.shape)
    
    MethylationData_in = MethylationData_in.join(OutcomeData_M, how='inner')
    MethylationData_in = MethylationData_in.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    #print('The shape of patients with race and outcome in methylation data is:')
    #print(MethylationData_in.shape)
    
    Data = MethylationData_in
    C = Data['C'].tolist()
    R = Data['race'].tolist()
    G = Data['G'].tolist()
    T = Data['T'].tolist()
    E = [1 - c for c in C]
    Data = Data.drop(columns=['C', 'race', 'T', 'G'])
    X = Data.values
    X = X.astype('float32')
    PackedData = {'X': X,
                  'T': np.asarray(T, dtype=np.float32),
                  'C': np.asarray(C, dtype=np.int32),
                  'E': np.asarray(E, dtype=np.int32),
                  'R': np.asarray(R),
                  'G': np.asarray(G),
                  'Samples': Data.index.values,
                  'FeatureName': list(Data)}
    
    return PackedData

def get_mRNA(cancer_type, target, groups):
    
    path = home_path + 'mRNAData/mRNA.mat'
    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['Y'], A['GeneName'][0], A['SampleName']
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]

    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    df_Y = df_Y[df_Y['Disease'].isin(tumor_types(cancer_type))]
    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns=['Disease'])

    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(cancer_type, df, target, groups)

def get_MicroRNA(cancer_type, target, groups):
    
    path = home_path + 'MicroRNAData/MicroRNA-Expression.mat'
    A = loadmat(path)
    X, Y, GeneName, SampleName = A['X'].astype('float32'), A['CancerType'], A['FeatureName'][0], A['SampleName']
    
    GeneName = [row[0] for row in GeneName]
    SampleName = [row[0][0] for row in SampleName]
    Y = [row[0][0] for row in Y]
    
    df_X = pd.DataFrame(X, columns=GeneName, index=SampleName)
    df_Y = pd.DataFrame(Y, index=SampleName, columns=['Disease'])
    df_Y = df_Y[df_Y['Disease'].isin(tumor_types(cancer_type))]
    df = df_X.join(df_Y, how='inner')
    df = df.drop(columns=['Disease'])

    index = df.index.values
    index_new = [row[:12] for row in index]
    df.index = index_new
    df = df.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')

    return add_race_CT(cancer_type, df, target, groups)

def add_race_CT(cancer_type, df, target, groups):
    
    df_race = get_race(cancer_type)
    df_race = df_race[df_race['race'].isin(groups)]
    df_C_T = get_CT(target)

    # Keep patients with race information
    df = df.join(df_race, how='inner')
    #print(df.shape)
    df = df.dropna(axis='columns')
    df = df.join(df_C_T, how='inner')
    #print(df.shape)

    # Packing the data
    C = df['C'].tolist()
    R = df['race'].tolist()
    G = df['G'].tolist()
    T = df['T'].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=['C', 'race', 'T', 'G'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X,
            'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32),
            'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R),
            'G': np.asarray(G),
            'Samples': df.index.values,
            'FeatureName': list(df)}

    return data

def get_fn(feature_type):
    
    fn = get_protein
    
    if feature_type == 'mRNA':
        fn = get_mRNA
        
    return fn

def get_dataset(cancer_type, feature_type, target, groups):
    
    fn = get_fn(feature_type)
    
    return fn(cancer_type, target=target, groups=groups)

def get_n_years(dataset, years):
    
    X, T, C, E, R, G = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R'], dataset['G']

    df = pd.DataFrame(X)
    df['T'] = T
    df['C'] = C
    df['R'] = R
    df['G'] = G
    df['Y'] = 1

    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    df['strat'] = df.apply(lambda row: str(row['Y']) + str(row['R']), axis=1)
    df['Gstrat'] = df.apply(lambda row: str(row['Y']) + str(row['G']), axis=1)
    df['GRstrat'] = df.apply(lambda row: str(row['G']) + str(row['Y']) + str(row['R']), axis=1)
    df = df.reset_index(drop=True)

    R = df['R'].values
    G = df['G'].values
    Y = df['Y'].values
    y_strat = df['strat'].values
    Gy_strat = df['Gstrat'].values
    GRy_strat = df['GRstrat'].values
    df = df.drop(columns=['T', 'C', 'R', 'G', 'Y', 'strat', 'Gstrat', 'GRstrat'])
    X = df.values
    y_sub = R # doese not matter

    return (X, Y.astype('int32'), R, y_sub, y_strat, G, Gy_strat, GRy_strat)

def normalize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    X = preprocessing.normalize(X)
    data_new['X'] = X
    
    return data_new

def standarize_dataset(data):
    
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new['X'] = X
    
    return data_new

def get_CT(target):
    
    path1 = home_path + 'TCGA-CDR-SupplementalTableS1.xlsx'
    cols = 'B,E,Z,AA'
    if target == 'DSS':
        cols = 'B,E,AB,AC'
    elif target == 'DFI':
        cols = 'B,E,AD,AE'
    elif target == 'PFI':
        cols = 'B,E,AF,AG'
    df_C_T = pd.read_excel(path1, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
    df_C_T.columns = ['G', 'E', 'T']
    df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    df_C_T['C'] = 1 - df_C_T['E']
    df_C_T.drop(columns=['E'], inplace=True)
    
    return df_C_T

def get_race(cancer_type):
    
    path = home_path + 'Genetic_Ancestry.xlsx'
    df_list = [pd.read_excel(path, disease, usecols='A,E', index_col='Patient_ID', keep_default_na=False)
               for disease in tumor_types(cancer_type)]
    df_race = pd.concat(df_list)
    df_race = df_race[df_race['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    df_race['race'] = df_race['EIGENSTRAT']

    df_race.loc[df_race['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    df_race.loc[df_race['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    df_race.loc[df_race['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    df_race.loc[df_race['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    df_race.loc[df_race['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    df_race = df_race.drop(columns=['EIGENSTRAT'])
    
    return df_race

def get_independent_data_single(dataset, query, groups):
    
    X, T, C, E, R, G = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R'], dataset['G']
    
    df = pd.DataFrame(X)
    df['R'] = R
    
    if query=='WHITE':
        mask = (df['R']==groups[0])
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
    elif query=='MG':
        mask = (df['R']==groups[1])
        X, T, C, E, R, G = X[mask], T[mask], C[mask], E[mask], R[mask], G[mask]
        
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R, 'G': G}
    
    return data

def merge_datasets(datasets):
    
    data = datasets[0]
    data = standarize_dataset(data)
    #print('Data has been standardized')
    X, T, C, E, R, G, Samples, FeatureName = data['X'], data['T'], data['C'], data['E'], data['R'], data['G'], data['Samples'], data['FeatureName']
    df = pd.DataFrame(X, index=Samples, columns=FeatureName)
    df['T'] = T
    df['C'] = C
    df['E'] = E
    df['R'] = R
    df['G'] = G

    for i in range(1, len(datasets)):
        data1 = datasets[i]
        data1 = standarize_dataset(data1)
        #print('Data has been standardized')
        X1, Samples, FeatureName = data1['X'], data1['Samples'], data1['FeatureName']
        temp = pd.DataFrame(X1, index=Samples, columns=FeatureName)
        df = df.join(temp, how='inner')

    # Packing the data and save it to the disk
    C = df['C'].tolist()
    R = df['R'].tolist()
    G = df['G'].tolist()
    T = df['T'].tolist()
    E = df['E'].tolist()
    df = df.drop(columns=['C', 'R', 'G', 'T', 'E'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X,
            'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32),
            'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R),
            'G': np.asarray(G),
            'Samples': df.index.values,
            'FeatureName': list(df)}

    return data

def run_cv_gender_race_comb(cancer_type, feature_type, target, groups):
    
    datasets = []
    for feature in feature_type:
        if feature=='Protein':
            print("==========================")
            print('fetching Protein data...')
            print("==========================")
            print(feature)
            Data = get_protein(cancer_type=cancer_type,target=target,groups=groups)
        if feature=='mRNA':
            print("==========================")
            print('fetching mRNA data...')
            print("==========================")
            print(feature)
            Data = get_mRNA(cancer_type=cancer_type,target=target,groups=groups)
        if feature=='MicroRNA':
            print("==========================")
            print('fetching MicroRNA data...')
            print("==========================")
            print(feature)
            Data = get_MicroRNA(cancer_type=cancer_type,target=target,groups=groups)
        if feature=='Methylation':
            print("==========================")
            print('fetching Methylation data...')
            print("==========================")
            print(feature)
            Data = get_Methylation(cancer_type=cancer_type,target=target,groups=groups)
        datasets.append(Data)
        
    return merge_datasets(datasets)






