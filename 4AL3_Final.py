# Authors: Karm Desai, John Mann
# Created: Sep 9, 2025
# Last Modified: Dec 4, 2025
# Purpose: This python includes SVM and RFC methods for a binary classifier to detect diabetes

# Dependencies: sklearn
# Python Version: 3.12

# References
# Sklearn Website - https://scikit-learn.org/stable/
# Python Documentation

#-------Imports-------#
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hinge_loss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import random

#-------Model Support Vector Machine (SVM) Definition-------#
class my_svm():
    def __init__(self, x_:list, y_:list):
        #Class features and labels initialization
        self.x = np.array(x_)
        self.y = np.array(y_)

        pass

    def preprocess(self):
        scalar = StandardScaler()
        self.x = scalar.fit_transform(self.x)
        
        return self.x, self.y
    
    def cross_validation(self, X, y, best_params):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tss_scores = []
        loss_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []

        TN_arr = []
        FP_arr = []
        FN_arr = []
        TP_arr = []
        cm_arr = []

        # K-fold cross-validation
        for train_idx, test_idx in kf.split(X, y):
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = self.training(X_train, y_train, best_params)

            y_pred = model.predict(X_test)
            tss_score, cm_arr = self.tss(y_test, y_pred)

            TN_arr.append(cm_arr[0])
            FP_arr.append(cm_arr[1])
            FN_arr.append(cm_arr[2])
            TP_arr.append(cm_arr[3])
            tss_scores.append(tss_score)

            loss_score = self.hinge_loss(y_test, y_pred)
            loss_scores.append(loss_score)

            precision, recall, f1 = self.f1_score(y_test, y_pred)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # calculate averages and confusion matrix entries
        cm_arr[0] = sum(TN_arr) / len(TN_arr)
        cm_arr[1] = sum(FP_arr) / len(FP_arr)
        cm_arr[2] = sum(FN_arr) / len(FN_arr)
        cm_arr[3] = sum(TP_arr) / len(TP_arr)

        avg_tss = sum(tss_scores) / len(tss_scores)
        avg_loss = sum(loss_scores) / len(loss_scores)

        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        return avg_tss, avg_loss, tss_scores, cm_arr, avg_precision, avg_recall, avg_f1, avg_accuracy
    
    def training(self, X_train, y_train, best_params):
        C_best = best_params['C']
        kernel_best = best_params['kernel']
        gamma_best = best_params.get('gamma', 'scale')

        model = SVC(kernel=kernel_best, C = C_best, class_weight='balanced', gamma=gamma_best)
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
        tss_score = true_positive - false_positive
        cm_arr = np.array([TN, FP, FN, TP])

        return tss_score, cm_arr
    
    def hinge_loss(self, y_true, y_pred):
        #change y_true, y_pred from 0 to -1 to use in hinge_loss function
        y_true_h = np.where(y_true == 0, -1, 1)
        y_pred_h = np.where(y_pred == 0, -1, 1)

        loss = hinge_loss(y_true_h, y_pred_h) # function uses -1 and 1 as classes
        return loss
    
    def f1_score(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division = 0)
        recall = recall_score(y_true, y_pred, zero_division = 0)
        f1 = f1_score(y_true, y_pred, zero_division = 0)
    
        return precision, recall, f1

#-------Model Random Forest Classifier (RFC) Definition-------#        
class my_RFC():
     def __init__(self, x_:list, y_:list):
        #Class features and labels initialization
        self.x = np.array(x_)
        self.y = np.array(y_)

        pass
     
     def preprocess(self):
        scalar = StandardScaler()
        self.x = scalar.fit_transform(self.x)
        
        return self.x, self.y
     
     def training(self, X_train, y_train):
        model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=None,
        random_state=42
        )

        model.fit(X_train, y_train)
        return model
     
     def hinge_loss(self, y_true, y_pred):
        #change y_true, y_pred from 0 to -1 to use in hinge_loss function
        y_true_h = np.where(y_true == 0, -1, 1)
        y_pred_h = np.where(y_pred == 0, -1, 1)

        loss = hinge_loss(y_true_h, y_pred_h) # function uses -1 and 1 as classes
        return loss
     
     def scores(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division = 0)
        recall = recall_score(y_true, y_pred, zero_division = 0)
        f1 = f1_score(y_true, y_pred, zero_division = 0)
    
        return precision, recall, f1
     
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
        tss_score = true_positive - false_positive
        cm_arr = np.array([TN, FP, FN, TP])

        return tss_score, cm_arr
     
     def cross_validation(self, X, y):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        loss_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []

        TN_arr = []
        FP_arr = []
        FN_arr = []
        TP_arr = []
        cm_arr = []

        # K-fold cross-validation
        for train_idx, test_idx in kf.split(X, y):
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = self.training(X_train, y_train)

            y_pred = model.predict(X_test)
            _, cm_arr = self.tss(y_test, y_pred)

            TN_arr.append(cm_arr[0])
            FP_arr.append(cm_arr[1])
            FN_arr.append(cm_arr[2])
            TP_arr.append(cm_arr[3])

            loss_score = self.hinge_loss(y_test, y_pred)
            loss_scores.append(loss_score)

            precision, recall, f1 = self.scores(y_test, y_pred)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # calculate averages and confusion matrix entries
        cm_arr[0] = sum(TN_arr) / len(TN_arr)
        cm_arr[1] = sum(FP_arr) / len(FP_arr)
        cm_arr[2] = sum(FN_arr) / len(FN_arr)
        cm_arr[3] = sum(TP_arr) / len(TP_arr)

        avg_loss = sum(loss_scores) / len(loss_scores)

        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

        return avg_loss, cm_arr, avg_precision, avg_recall, avg_f1, avg_accuracy

def boyer_moore(y_vals):
    N = len(y_vals)
    output = -1
    unique_arr, count_arr = np.unique(y_vals, return_counts=True)
    for i in range(0, len(unique_arr) - 1):
        if count_arr[i] > N/2:
            output = unique_arr[i]
    
    return output


#------Experiment on Models------#
def experiment():

    #--Dataset Retrieval--#
    diabetes_data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", nrows=1000)
    ddf = diabetes_data.drop(columns=["AnyHealthcare", "NoDocbcCost", "MentHlth", "DiffWalk", "Education", "Income"])

    y_vals = ddf['Diabetes_binary'].values
    features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Sex', 'Age', "CholCheck", "GenHlth", "PhysHlth"]

    #--Array Initialization--#
    X_vals = [] # Features array
    all_tss = [] # TSS values
    cm_arr = [] # Confusion Matrix values

    #--Append feature values to feature array--#
    for name in features:
        X_vals.append(ddf[name].values)

    #--Baseline Model Prediction--#
    boyer_moore_out = boyer_moore(y_vals)
    print("Baseline Prediction: ", boyer_moore_out)

    #--Feature Vector Selection--#
    # Convert features to matrix (same as before)
    X_vals = np.column_stack([ddf[name].values for name in features])
    selector = SelectKBest(score_func=f_classif, k=10) 
    X_selected = selector.fit_transform(X_vals, y_vals)
    selected_mask = selector.get_support()
    clipped_features = [feat for feat, keep in zip(features, selected_mask) if keep]
    clipped_X_vals = X_selected
    print("Selected features:", clipped_features)
 
    #--SVM Model Creation--#
    svm_model = my_svm(clipped_X_vals, y_vals)
    
    X_preprocessed_svm, y_preprocessed_svm = svm_model.preprocess()

    #--GridSearchCV for Hyperparameter Tuning--#
    grid_params = {
        'C': [0.1, 1, 10, 50, 100, 300, 500, 1000],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    svc_test = SVC()
    grid_search = GridSearchCV(svc_test, grid_params, scoring='accuracy', cv=5)
    grid_search.fit(X_preprocessed_svm, y_preprocessed_svm)
    best_params = grid_search.best_params_
    print("Best parameters from GridSearchCV: ", best_params)
    
    #--Cross Validation SVM--#
    avg_tss, avg_loss, all_tss, cm_arr, avg_precision, avg_recall, avg_f1, avg_acc = svm_model.cross_validation(X_preprocessed_svm, y_preprocessed_svm, best_params)

    #--RFC Model Creation--#
    rfc_model = my_RFC(clipped_X_vals, y_vals)
    X_preprocessed_rfc, y_preprocessed_rfc = rfc_model.preprocess()

    #--Cross Validation RFC--#
    rfc_avg_loss, rfc_cm_arr, rfc_avg_precision, rfc_avg_recall, rfc_avg_f1, rfc_avg_acc = rfc_model.cross_validation(X_preprocessed_rfc, y_preprocessed_rfc)

    #--SVM Results--#
    print("SVM Training Results:")
    print("avg_tss: ", avg_tss)
    print("avg_loss: ", avg_loss)
    print("avg_precision: ", avg_precision)
    print("avg_recall: ", avg_recall)
    print("avg_f1: ", avg_f1)
    print("avg_acc: ", avg_acc)
    print("---------------------------------")

    #--RFC Results--#
    print("RFC Training Results:")
    print("avg_loss: ", rfc_avg_loss)
    print("avg_precision: ", rfc_avg_precision)
    print("avg_recall: ", rfc_avg_recall)
    print("avg_f1: ", rfc_avg_f1)
    print("avg_acc: ", rfc_avg_acc)     

    #--Plotting--#
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    mean = np.mean(all_tss)
    std_dev = np.std(all_tss)

    # Confusion Matrix SVM
    cm = np.array([[cm_arr[0], cm_arr[1]], [cm_arr[2], cm_arr[3]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    disp.ax_.set_title(f"Confusion Matrix SVM")

    # Confusion Matrix RFC
    cm2 = np.array([[rfc_cm_arr[0], rfc_cm_arr[1]], [rfc_cm_arr[2], rfc_cm_arr[3]]])
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=[0,1])
    disp2.plot()
    disp2.ax_.set_title(f"Confusion Matrix RFC")

    # TSS Scores Graph SVM
    ax1.bar(range(1, len(all_tss)+1), all_tss)
    ax1.set_xlabel("K-Fold")
    ax1.set_ylabel("TSS Score")
    ax1.set_ylim(-1,1)
    ax1.set_title(f"TSS Scores per K-Fold \nGiven:\n Mean (Avg. TSS): {mean:.4f} \u03C3: {std_dev:.4f}")

    '''
    # View Correlation on Bar Graph
    ax2.bar(features, correlation_arr)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Correlation Score")
    ax2.set_ylim(-1,1)
    ax2.set_title(f"Correlation Between Features and Outputs")
    '''
    plt.show()
     
experiment()