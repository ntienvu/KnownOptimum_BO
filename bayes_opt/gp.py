# -*- coding: utf-8 -*-
"""
Created on April 2020

@author: Vu Nguyen
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import scipy
#from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
#import matplotlib as mpl
import matplotlib.cm as cm

class GaussianProcess(object):
    def __init__ (self,SearchSpace,noise_delta=1e-8,verbose=0):
        self.noise_delta=noise_delta
        self.noise_upperbound=noise_delta
        self.mycov=self.cov_RBF
        self.SearchSpace=SearchSpace
        scaler = MinMaxScaler()
        scaler.fit(SearchSpace.T)
        self.Xscaler=scaler
        self.verbose=verbose
        self.dim=SearchSpace.shape[0]
        
        self.hyper={}
        self.hyper['var']=1 # standardise the data
        self.hyper['lengthscale']=0.04 #to be optimised
        self.noise_delta=noise_delta
        return None
        
    def set_optimum_value(self,fstar_scaled):
        self.fstar=fstar_scaled
        
    def fit(self,X,Y,IsOptimize=0):
        """
        Fit a Gaussian Process model
        X: input 2d array [N*d]
        Y: output 2d array [N*1]
        """       
        self.X_ori=X # this is the output in original scale
        #self.X= self.Xscaler.transform(X) #this is the normalised data [0-1] in each column
        self.X=X
        self.Y_ori=Y # this is the output in original scale
        self.Y=(Y-np.mean(Y))/np.std(Y) # this is the standardised output N(0,1)
        
        if IsOptimize:
            self.hyper['lengthscale']=self.optimise()         # optimise GP hyperparameters
            
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta     
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x !")
      
        self.L=scipy.linalg.cholesky(self.KK_x_x,lower=True)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        
    def cov_RBF(self,x1, x2,hyper):        
        """
        Radial Basic function kernel (or SE kernel)
        """
        variance=hyper['var']
        lengthscale=hyper['lengthscale']

        if x1.shape[1]!=x2.shape[1]:
            x1=np.reshape(x1,(-1,x2.shape[1]))
        Euc_dist=euclidean_distances(x1,x2)

        return variance*np.exp(-np.square(Euc_dist)/lengthscale)
    

    def log_llk(self,X,y,hyper_values):
        
        #print(hyper_values)
        hyper={}
        hyper['var']=1
        hyper['lengthscale']=hyper_values[0]
        noise_delta=self.noise_delta

        KK_x_x=self.mycov(X,X,hyper)+np.eye(len(X))*noise_delta     
        if np.isnan(KK_x_x).any(): #NaN
            print("nan in KK_x_x !")   

        try:
            L=scipy.linalg.cholesky(KK_x_x,lower=True)
            alpha=np.linalg.solve(KK_x_x,y)

        except: # singular
            return -np.inf
        try:
            first_term=-0.5*np.dot(self.Y.T,alpha)
            W_logdet=np.sum(np.log(np.diag(L)))
            second_term=-W_logdet

        except: # singular
            return -np.inf

        logmarginal=first_term+second_term-0.5*len(y)*np.log(2*3.14)
        
        #print(hyper_values,logmarginal)
        return np.asscalar(logmarginal)
    
    def set_ls(self,lengthscale):
        self.hyper['lengthscale']=lengthscale
        
    def optimise(self):
        """
        Optimise the GP kernel hyperparameters
        Returns
        x_t
        """
        opts ={'maxiter':200,'maxfun':200,'disp': False}

        # epsilon, ls, var, noise var
        #bounds=np.asarray([[9e-3,0.007],[1e-2,self.noise_upperbound]])
        bounds=np.asarray([[1e-3,1]])

        init_theta = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(10, 1))
        logllk=[0]*init_theta.shape[0]
        for ii,val in enumerate(init_theta):           
            logllk[ii]=self.log_llk(self.X,self.Y,hyper_values=val) #noise_delta=self.noise_delta
            
        x0=init_theta[np.argmax(logllk)]

        res = minimize(lambda x: -self.log_llk(self.X,self.Y,hyper_values=x),x0,
                                   bounds=bounds,method="L-BFGS-B",options=opts)#L-BFGS-B
        
        if self.verbose:
            print("estimated lengthscale",res.x)
            
        return res.x  
   
    def predict(self,Xtest,isOriScale=False):
        """
        ----------
        Xtest: the testing points  [N*d]

        Returns
        -------
        pred mean, pred var, pred mean original scale, pred var original scale
        """    
        
        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)
            
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
       
        KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_x=self.mycov(Xtest,self.X,self.hyper)

        mean=np.dot(KK_xTest_x,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_x.T)
        var=KK_xTest_xTest-np.dot(v.T,v)

        #mean_ori=mean*np.std(self.Y_ori)+np.mean(self.Y_ori)
        std=np.reshape(np.diag(var),(-1,1))
        
        #std_ori=std*np.std(self.Y_ori)#+np.mean(self.Y_ori)
        
        #return mean,std,mean_ori,std_ori
        return  np.reshape(mean,(-1,1)),std  
