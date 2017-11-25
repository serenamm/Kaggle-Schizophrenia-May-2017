'''
Final project for MATH 6050A, Spring 2016/17.
'''

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from helper import load_data
import numpy as np
from time import time
import csv
from itertools import izip

if __name__ == "__main__":
    ids, data, labels = load_data()
    
    ifPCA = True
    ifCV = True
    ifTest = True
    
    t0 = time()
    
    if ifPCA:
        print("Conduct PCA.")
        n_components = 16;
        pca = PCA(n_components,svd_solver='randomized',whiten=True).fit(data)
        data = pca.transform(data)
        exrat = 100*pca.explained_variance_ratio_.sum()
        print("Use %d principal components which explains %3.2f %% variance." % (n_components,exrat) )
    
    if ifCV:
        print("Conduct Model Selection.")
        C_range = np.logspace(-1,4,num=8)
        gamma_range = np.logspace(-4,1,num=8)
        param_grid = dict(gamma=gamma_range, C=C_range)
        rf = svm.SVC(kernel='rbf')
        random_state = 1
        gs = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5, n_jobs=-1,verbose=1)
        gs = gs.fit(data,labels)

        print(gs.best_params_)

        bp = gs.best_params_
        random_state = 1
        clf = svm.SVC(probability=True,kernel='rbf',C=bp['C'],gamma=bp['gamma'],verbose=True).fit(data,labels)    
    else:
        random_state = 1
        clf = svm.SVC(probability=True,verbose=True).fit(data,labels) 

    print("Training is done in %1.3f s." % (time()-t0))
    
    if ifTest:
        print("Test begins.")    
        testids, testdata = load_data(False)
    
        if ifPCA:
            testdata = pca.transform(testdata)
    
        random_state = 1
        preds = clf.predict_proba(testdata)[:,1]

        print("Write submission file.")
        with open("pysubmission.csv",'w') as f:
            w = csv.writer(f)
            w.writerow(["ID","Probability"])
            for item in izip(testids, preds):
                w.writerow(item)
            
        print("Submission file is completed.")