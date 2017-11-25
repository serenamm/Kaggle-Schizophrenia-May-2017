# Kaggle-Schizophrenia-May-2017

# Abstract

This was a project for MATH 6450A, Statistical Machine Learning, completed with my partner Yi-Su Lo. Using a dataset from Kaggle, we attempted to automatically diagnose patients with schizophrenia. After performing PCA and model selection, we found that scikit-learn’s naïve SVM was sufficient to place us 22nd in the competition, on the private leaderboard. 

# Skills

This demonstrates preprocessing, model selection (using grid search and cross validation) and machine learning skills in Python, using scikit-learn.

# Data

The data is from Kaggle’s MLSP 2014 Schizophrenia Classification Challenge (https://www.kaggle.com/c/mlsp-2014-mri), and is split into train and test sets. The datasets is made up of features derived from MRI scans of healthy and unhealthy individuals. Specifically, there are two types of features in the datasets are functional network connectivity (FNC) and source based morphometry (SBM) features. FNC features are correlation values that describe the connectivity between different structures of the brain overtime. SBM features are standardized weights that the describe expression of brain maps, which are maps denoting independent gray matter structures in the brain.

# Method

The algorithm we applied to the problem is as follows:
1.	Load and process the data.
2.	Principle component analysis to reduce the dimension of the data points (optional).
3.	Perform model selection using grid search and 5-fold cross validation to determine appropriate parameters for training.
4.	Use support vector machine (SVM) with the Gaussian radial basis function to train a classifier.
5.	Make predictions on the test set with the trained SVM classifier, and transform the classification score into a real number in [0,1] as the probability of illness.

# Results and Conclusion

We used AUC (ROC) as our error metric, as specified in the Kaggle competition. We were surprised to find that our best score was obtained with scikit-learn’s naïve SVM classifier, using 32 principle components, with an 8x8 mesh for the model selection, which provided better results than a 16x16 mesh. Note that once a Kaggle competition closes, participants are able to see their scores on the “Leaderboards” immediately. Our best score was 0.90179 on the public board, and 0.84615 on the private board, placing us 28th and 104th, respectively. However, using naïve SVM, no PCA, and an 8x8 mesh would have placed us 89th on the public board, and 22nd on the private board, with AUC of 0.85714 and 0.88718, respectively.


