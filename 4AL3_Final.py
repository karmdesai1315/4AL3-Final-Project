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
    def __init__(self, x_:list, y_:list):
        #Class features and labels initialization
        self.x = np.array(x_)
        self.y = np.array(y_)

        pass

    def preprocess(self):
        #Removing nan values from CSV file if they exist
        '''
        mask = ~np.isnan(self.x).any(axis=1) & ~np.isnan(self.y)
        self.x = self.x[mask]
        self.y = self.y[mask]
        '''

        '''
        #Call standardscaler for normalization
        scalar = StandardScaler()

        #Preprocess self.x and rewrite to same variable
        self.x = scalar.fit_transform(self.x)

        '''
        scalar = StandardScaler()
        self.x = scalar.fit_transform(self.x)
        return self.x, self.y
    
    def cross_validation(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        tss_scores = []
        loss_scores = []
        TN_arr = []
        FP_arr = []
        FN_arr = []
        TP_arr = []
        cm_arr = []

        for train_idx, test_idx in kf.split(X):
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = self.training(X_train, y_train)

            y_pred = model.predict(X_test)
            print(y_pred)
            print(y_test)
            tss_score, cm_arr = self.tss(y_test, y_pred)
            TN_arr.append(cm_arr[0])
            FP_arr.append(cm_arr[1])
            FN_arr.append(cm_arr[2])
            TP_arr.append(cm_arr[3])
            tss_scores.append(tss_score)
            loss_score = self.hinge_loss(y_test, y_pred)
            loss_scores.append(loss_score)

        cm_arr[0] = sum(TN_arr) / len(TN_arr)
        cm_arr[1] = sum(FP_arr) / len(FP_arr)
        cm_arr[2] = sum(FN_arr) / len(FN_arr)
        cm_arr[3] = sum(TP_arr) / len(TP_arr)
        avg_tss = sum(tss_scores) / len(tss_scores)
        return avg_tss, loss_score, tss_scores, cm_arr
    
    def training(self, X_train, y_train):
        model = SVC(kernel="rbf", C = 1)
        model.fit(X_train, y_train)
        return model

    def tss(self, y_true, y_pred):
        
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        if TP + FN > 0:
            true_positive = TP/(TP + FN)
        else:
            true_positive = 0
        
        if FP + TN > 0:
            false_positive = FP/(FP + TN)
        else:
            false_positive = 0
        print("TP: ", TP)
        print("TN: ", TN)
        print("FP: ", FP)
        print("FN: ", FN)
        tss_score = true_positive - false_positive
        cm_arr = np.array([TN, FP, FN, TP])

        return tss_score, cm_arr
    
    def hinge_loss(self, y_true, y_pred):
        
        loss = np.maximum(0, 1 - y_true * y_pred)

        return np.mean(loss)

def experiment():
    diabetes_data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", nrows=1000)
    ddf = diabetes_data.drop(columns=["CholCheck","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Education", "Income"])

    y_vals = ddf['Diabetes_binary'].values
    features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Sex', 'Age']
    #X_vals = np.column_stack([ddf[name].values for name in features])
    #features = ['Stroke']
    X_vals = []
    all_tss = []
    cm_arr = []

    for name in features:
        X_vals.append(ddf[name].values)

    #print(X_vals[0])
    X_vals = np.column_stack(X_vals)
    print(X_vals[0])
    print(y_vals)

    svm_model = my_svm(X_vals, y_vals)
    X_preprocessed, y_preprocessed = svm_model.preprocess()
    print("Shape of X:", X_preprocessed.shape)
    print("Shape of y:", y_preprocessed.shape)
    print("First 5 labels:", y_preprocessed[:5])
    
    avg_tss, avg_loss, all_tss, cm_arr = svm_model.cross_validation(X_preprocessed, y_preprocessed)

    #print("X_pre: ",X_preprocessed)
    #print("Y_pre: ",y_preprocessed)

    print("avg_tss: ", avg_tss)
    print("avg_loss: ", avg_loss)

    mean = np.mean(all_tss)
    std_dev = np.std(all_tss)
    #Displaying all confusion matricies and bar graphs
    cm = np.array([[cm_arr[0], cm_arr[1]], [cm_arr[2], cm_arr[3]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    disp.ax_.set_title(f"Confusion Matrix")

    plt.figure(figsize=(8,4))
    plt.bar(range(1, len(all_tss)+1), all_tss)
    plt.xlabel("K-Fold")
    plt.ylabel("TSS Score")
    plt.ylim(-1,1)
    plt.title(f"TSS Scores per K-Fold \nGiven:\n Mean (Avg. TSS): {mean:.4f} \u03C3: {std_dev:.4f}")
    plt.show()
    
experiment()