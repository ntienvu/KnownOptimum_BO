# -*- coding: utf-8 -*-


import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from bayes_opt.test_functions.drl.vdrl_pg_a2c_cartpole_blackbox import evaluate,evaluate_with_maxiter
from bayes_opt.test_functions.drl.vdrl_dueling_dqn_blackbox_per import evaluate_dueling_dqn_with_maxiter

from bayes_opt.test_functions.drl.vdrl_pg_a2c_cartpole_blackbox_exp_replay import evaluateER

def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x

class DRL_Cartpole_A2C:

    '''
    DRL_Cartpole_A2C 
    
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('gamma',(0.9,1)),('lr_pm',(0.00001,0.01)),('lr_vm',(0.00001,0.01))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 200
        self.ismax=1
        self.name='DRL_Cartpole_A2C'
        

    
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
    
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Reward=evaluate(X)
        else:
            Reward=np.apply_along_axis(evaluate,1,X)

        #print RMSE    
        return Reward*self.ismax  
    
class DRL_Cartpole_A2C_MaxIter:

    '''
    DRL_Cartpole_A2C 
    with MaxIteration input
    
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            #self.bounds = OrderedDict([('gamma',(0.9,1)),('lr_pm',(0.00001,0.01)),('lr_vm',(0.00001,0.01))])
            self.bounds = OrderedDict([('gamma',(0.8,1)),('lr_pm',(0.000001,0.02)),('lr_vm',(0.000001,0.02))])

        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 200
        self.ismax=1
        self.name='DRL_Cartpole_A2C_MaxIter'
        

    
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
    
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Reward=evaluate_with_maxiter(X)
        else:
            Reward=np.apply_along_axis(evaluate_with_maxiter,1,X)

        #print RMSE    
        return Reward*self.ismax  

    
class DRL_Cartpole_A2C_ExpR:

    '''
    DRL_Cartpole_A2C 
    
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('gamma',(0.9,1)),('lr_pm',(0.00001,0.01)),('lr_vm',(0.00001,0.01))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fstar = 200
        self.ismax=1
        self.name='DRL_Cartpole_A2C_ExpR'
        

    
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
    
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Reward=evaluateER(X)
        else:
            Reward=np.apply_along_axis(evaluateER,1,X)

        #print RMSE    
        return Reward*self.ismax  