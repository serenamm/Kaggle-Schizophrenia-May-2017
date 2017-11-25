'''
Final project for MATH 6050A, Spring 2016/17.
'''

import csv
import numpy as np

def load_data(train=True, sbm_only=False, fnc_only=False):
    if train:
        fnc="Train/train_FNC.csv"
        sbm="Train/train_SBM.csv"
    else:
        fnc="Test/test_FNC.csv"
        sbm="Test/test_SBM.csv"
        
    with open(fnc,'r') as f:
        train_fnc = list(csv.reader(f))
    fnc_header = train_fnc[0]
    fnc_data = np.array([np.array(map(float,i)) for i in train_fnc[1:]])
    ids = np.array(fnc_data[:,0],dtype=int)

    with open(sbm,'r') as f:
        train_sbm = list(csv.reader(f))
    sbm_header = train_sbm[0]
    sbm_data = np.array([np.array(map(float,i)) for i in train_sbm[1:]])
    fnc_data = fnc_data[:,1:]
    sbm_data = sbm_data[:,1:]
    data = np.column_stack((sbm_data,fnc_data))
    #data = np.column_stack((fnc_data,sbm_data))
    
    if not train:
        return ids, data

    with open("Train/train_labels.csv",'r') as f:
        f.next()
        labels = np.array([int(i[1]) for i in csv.reader(f)])

    if sbm_only:
        return ids,sbm_data,labels
    elif fnc_only:
        return ids,fnc_data,labels
    else:
        return ids, data, labels