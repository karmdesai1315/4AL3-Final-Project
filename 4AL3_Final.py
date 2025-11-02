import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import random

class my_svm():
    def __init__(self, x_:list, y_:list, data_set):
        

        self.input = np.array(x_)
        self.target = np.array(y_)
        self.data = data_set
        
        pass
#############################################################

    def preprocess(self,):

        #Removing zeros from passed data
        non_zeros = ~np.isnan(self.input).any(axis=1) & ~np.isnan(self.target)
        self.input = self.input[non_zeros]
        self.target = self.target[non_zeros]

        #Using StandardScaler for preprocessing
        scaler = StandardScaler()

        #Updating self.input
        self.input = scaler.fit_transform(self.input)

        return self.input, self.target
#############################################################

    def feature_creation(self, fs_value):
        # CODE HERE !
        data_set = self.data
        pos_data_set = []
        neg_data_set = []
        
        #loading all data sets and creating features based on the fs value passed
        tc_pos = np.load(f'{data_set}\pos_features_main_timechange.npy', mmap_mode='r', allow_pickle=True)
        tc_neg = np.load(f'{data_set}\\neg_features_main_timechange.npy', mmap_mode='r', allow_pickle=True)
        mm_pos = np.load(f'{data_set}\pos_features_maxmin.npy', mmap_mode='r', allow_pickle=True)
        mm_neg = np.load(f'{data_set}\\neg_features_maxmin.npy', mmap_mode='r', allow_pickle=True)
        his_pos = np.load(f'{data_set}\pos_features_historical.npy', mmap_mode='r', allow_pickle=True)
        his_neg  = np.load(f'{data_set}\\neg_features_historical.npy', mmap_mode='r', allow_pickle=True)
        
        for fs in fs_value:
            if fs == "FS-I":
                X = tc_pos[:, [0, 17]]
                Y = tc_neg[:, [0, 17]]
            elif fs == "FS-II":
                X = tc_pos[:, [18, 89]]
                Y = tc_neg[:, [18, 89]]
            elif fs == "FS-III":
                X = his_pos[:, [0]]
                Y = his_neg[:, [0]]
            else:
                X = mm_pos[:, [0, 17]]
                Y = mm_neg[:, [0, 17]]
            pos_data_set.append(X)
            neg_data_set.append(Y)

        set_complete = np.vstack((np.hstack(pos_data_set), np.hstack(neg_data_set)))

        return set_complete
#############################################################

    def cross_validation(self, X, Y):

        scores = []
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_data, test_data in kf.split(X):
            X_train, X_test = X[train_data], X[test_data]
            Y_train, Y_test = Y[train_data], Y[test_data]
            
            trained = self.training(X_train, Y_train)

            Y_pred = trained.predict(X_test)

            #TSS function call
            score = self.tss(Y_test, Y_pred)
            scores.append(score)
        
        avg_scores = np.mean(scores)

        return avg_scores
#############################################################


    #training() function trains a SVM classification model on input features and corresponding target
    def training(self,X, Y):
        #Manually tested kernel solution
        sol = SVC(kernel="rbf")
        sol.fit(X, Y)
        
        return sol
#############################################################


    def tss(self,Y_T, Y_P):

        TP = np.sum((Y_T == 1) & (Y_P == 1))
        TN = np.sum((Y_T == 0) & (Y_P == 0))
        FP = np.sum((Y_T == 0) & (Y_P == 1))
        FN = np.sum((Y_T == 1) & (Y_P == 0))

        #Div by 0 handling
        if TP + FN > 0:
            form1 = TP/(TP + FN)
        else:
            form1 = 0
        if FP + TN > 0:
            form2 = FP/(FP + TN)

        else:
            form2 = 0
        
        skill_score = form1 - form2
        ###########
        return skill_score
    

def experiment():
    var = ''