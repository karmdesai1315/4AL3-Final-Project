import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hinge_loss
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.datasets import make_classification
import random


class my_svm():
    def __init__(self, x_:list, y_:list):
        #Class features and labels initialization
        self.x = np.array(x_)
        self.y = np.array(y_)

        pass

    def preprocess(self):
        #Removing nan values from CSV file if they exist
        scalar = StandardScaler()
        self.x = scalar.fit_transform(self.x)
        
        return self.x, self.y
    
    def cross_validation(self, X, y, best_params):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        #kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #kf = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.2, random_state=42)
        #kf.get_n_splits(X, y)
        tss_scores = []
        loss_scores = []
        TN_arr = []
        FP_arr = []
        FN_arr = []
        TP_arr = []
        cm_arr = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_idx, test_idx in kf.split(X):
            
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


        cm_arr[0] = sum(TN_arr) / len(TN_arr)
        cm_arr[1] = sum(FP_arr) / len(FP_arr)
        cm_arr[2] = sum(FN_arr) / len(FN_arr)
        cm_arr[3] = sum(TP_arr) / len(TP_arr)

        avg_tss = sum(tss_scores) / len(tss_scores)
        avg_loss = sum(loss_scores) / len(loss_scores)

        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)

        return avg_tss, avg_loss, tss_scores, cm_arr, avg_precision, avg_recall, avg_f1
    
    def training(self, X_train, y_train, best_params):
        C_best = best_params['C']
        kernel_best = best_params['kernel']
        gamma_best = best_params.get('gamma', 'scale')
        #model = SVC(kernel="rbf", C = 0.1, class_weight='balanced', gamma='scale')
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
        #print("TP: ", TP)
        #print("TN: ", TN)
        #print("FP: ", FP)
        #print("FN: ", FN)
        tss_score = true_positive - false_positive
        cm_arr = np.array([TN, FP, FN, TP])

        return tss_score, cm_arr
    
    def hinge_loss(self, y_true, y_pred):
        #change y_true, y_pred from 0 to -1 to use in hinge_loss function
        for i in range(len(y_true)):
            if y_true[i] == 0:
                y_true[i] = -1
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                y_pred[i] = -1
        #loss = np.maximum(0, 1 - y_true * y_pred)
        loss = hinge_loss(y_true, y_pred) # function uses -1 and 1 as classes
        return loss
    
    def f1_score(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division = 0)
        recall = recall_score(y_true, y_pred, zero_division = 0)
        f1 = f1_score(y_true, y_pred, zero_division = 0)
    
        return precision, recall, f1
        
def boyer_moore(y_vals):
    N = len(y_vals)
    output = -1
    unique_arr, count_arr = np.unique(y_vals, return_counts=True)
    for i in range(0, len(unique_arr) - 1):
        if count_arr[i] > N/2:
            output = unique_arr[i]
    
    return output


def experiment():
    diabetes_data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", nrows=1000)
    #ddf = diabetes_data.drop(columns=["CholCheck","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Education", "Income"])
    ddf = diabetes_data

    y_vals = ddf['Diabetes_binary'].values
    #features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Sex', 'Age']
    features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'Sex', 'Age', "CholCheck","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Education", "Income"]
    #X_vals = np.column_stack([ddf[name].values for name in features])
    #features = ['Stroke']
    X_vals = []
    all_tss = []
    cm_arr = []

    correlation_arr = []

    boyer_moore_out = boyer_moore(y_vals)

    for name in features:
        X_vals.append(ddf[name].values)

    #---------Feature Correlation---------#
    for arr in X_vals:
        r = np.corrcoef(arr, y_vals)
        correlation_arr.append(r[0,1])

    X_vals = np.column_stack(X_vals)
    #print(correlation_arr)

    clipped_features = []
    for i in range(1, len(features)):
        if correlation_arr[i] > 0.1:
            clipped_features.append(features[i])

    clipped_X_vals = []
    for name in clipped_features:
        clipped_X_vals.append(ddf[name].values)

    print(clipped_features)
    
    clipped_X_vals = np.column_stack(clipped_X_vals)

    #------View Correlation on Bar Graph------#
    # Create a bar graph
    '''
    corr_target_graph = plt.bar(features, correlation_arr, color="skyblue")

    # Set the title and change the font size of y labels
    corr_target_graph.title('Correlation with Diabetes_binary', fontsize=12)
    corr_target_graph.tick_params(axis='y', labelsize=8)

    # Remove the spines of top, left and right
    corr_target_graph.spines[['top', 'left', 'right']].set_visible(False)
    '''
    
    #----------SVM Model Creation-------#
    svm_model = my_svm(clipped_X_vals, y_vals)
    X_preprocessed, y_preprocessed = svm_model.preprocess()
    
    grid_params = {
        'C': [0.1, 1, 10, 50],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    svc_test = SVC()
    grid_search = GridSearchCV(svc_test, grid_params, scoring='accuracy', cv=5)
    grid_search.fit(X_preprocessed, y_preprocessed)
    best_params = grid_search.best_params_
    print("Best parameters from GridSearchCV: ", best_params)

    avg_tss, avg_loss, all_tss, cm_arr, avg_precision, avg_recall, avg_f1 = svm_model.cross_validation(X_preprocessed, y_preprocessed, best_params)

    #print("X_pre: ",X_preprocessed)
    #print("Y_pre: ",y_preprocessed)

    print("avg_tss: ", avg_tss)
    print("avg_loss: ", avg_loss)
    print("avg_precision", avg_precision)
    print("avg_recall", avg_recall)
    print("avg_f1", avg_f1)

    print("Baseline Prediction: ", boyer_moore_out)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    mean = np.mean(all_tss)
    std_dev = np.std(all_tss)
    #Displaying all confusion matricies and bar graphs
    cm = np.array([[cm_arr[0], cm_arr[1]], [cm_arr[2], cm_arr[3]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    disp.ax_.set_title(f"Confusion Matrix")

    #ax1.figure(figsize=(8,4))
    ax1.bar(range(1, len(all_tss)+1), all_tss)
    ax1.set_xlabel("K-Fold")
    ax1.set_ylabel("TSS Score")
    ax1.set_ylim(-1,1)
    ax1.set_title(f"TSS Scores per K-Fold \nGiven:\n Mean (Avg. TSS): {mean:.4f} \u03C3: {std_dev:.4f}")

    ax2.bar(features, correlation_arr)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Correlation Score")
    ax2.set_ylim(-1,1)
    ax2.set_title(f"Correlation Between Features and Outputs")
    plt.show()

    
experiment()