# -*- coding: utf-8 -*-



import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR
import math
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import gzip

#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
from sklearn.metrics import f1_score 
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def plot(self):
        print("not implemented")
        

class XGBoost:

    '''
    XGBoost: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 20)),('colsample_bytree',(0.1, 1)),('max_depth',(5,15)),('subsample',(0.5,1)),
                                        ('gamma',(0,10)),('alpha',(0,10))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 0.804
        self.ismax=1
        self.name='XGBoost_Classification'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
        
    def run_XGBoost(self,X):
        #print(X)
        params={}
        params['min_child_weight'] = int(X[0])
        params['colsample_bytree'] = max(min(X[1], 1), 0)
        params['max_depth'] = int(X[2])
        params['subsample'] = max(min(X[3], 1), 0)
        params['gamma'] = max(X[4], 0)
        params['alpha'] = max(X[5], 0)
        #params['silent'] = 1
    
        #print(params)
    
        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        # make predictions for test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
   
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        from numpy import loadtxt
    
        dataset = loadtxt('P:/05.BayesianOptimization/PradaBayesianOptimization/prada_bayes_opt/test_functions/data/pima-indians-diabetes.csv', delimiter=",")
 
        # split data into X and y
        Xdata = dataset[:,0:8]
        Ydata = dataset[:,8]
        
        # split data into train and test sets
        seed = 1
        test_size = 0.5
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xdata, Ydata, test_size=test_size, random_state=seed)
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_XGBoost(X)
        else:
            Accuracy=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return Accuracy*self.ismax 
    

class XGBoost_Skin:

    '''
    XGBoost: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('min_child_weight',(1, 20)),('colsample_bytree',(0.1, 1)),('max_depth',(5,15)),('subsample',(0.5,1)),
                                        ('gamma',(0,10)),('alpha',(0,10))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 1
        self.ismax=1
        self.name='XGBoost_Skin_Classification'
        self.X_train=None
        self.X_test=None
        self.Y_train=None
        self.Y_test=None
        
        from numpy import loadtxt
    
        #dataset = loadtxt('P:/05.BayesianOptimization/PradaBayesianOptimization/prada_bayes_opt/test_functions/data/pima-indians-diabetes.csv', delimiter=",")
        dataset = loadtxt('bayes_opt/test_functions/data/Skin_NonSkin.txt', delimiter="\t")
 
        # split data into X and y
        Xdata = dataset[:,0:3]
        Ydata = dataset[:,3]
        
        # split data into train and test sets
        seed = 2
        test_size = 0.85
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xdata, Ydata, test_size=test_size, random_state=seed)
        
        
    def run_XGBoost(self,X):
        #print(X)
        params={}
        params['min_child_weight'] = int(X[0])
        params['colsample_bytree'] = max(min(X[1], 1), 0)
        params['max_depth'] = int(X[2])
        params['subsample'] = max(min(X[3], 1), 0)
        params['gamma'] = max(X[4], 0)
        params['alpha'] = max(X[5], 0)
        #params['silent'] = 1
    
        #print(params)
    
        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        # make predictions for test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
   
    
    def func(self,X):
        X=np.asarray(X)

        np.random.seed(1)  # for reproducibility
        
        #import pandas as pd
        #from sklearn.preprocessing import LabelEncoder
        #from tqdm import tqdm
        
        #print(X)
        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_XGBoost(X)
        else:
            Accuracy=np.apply_along_axis( self.run_XGBoost,1,X)

        #print RMSE    
        return Accuracy*self.ismax 
    
    
 