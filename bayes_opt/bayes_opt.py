# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:51:04 2020

@author: Lenovo
"""



import numpy as np
#from gp import GaussianProcess
import matplotlib.pyplot as plt

import time
from sklearn.preprocessing import MinMaxScaler
from bayes_opt.gp import GaussianProcess
from bayes_opt.utilities import acq_max_with_name


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BayesOpt:

    def __init__(self, func, SearchSpace,acq_name="ei",verbose=1):
        """      
        Input parameters
        ----------
        
        func:                       a function to be optimized
        SearchSpace:                bounds on parameters        
        acq_name:                   acquisition function name, such as [ei, gp_ucb]
                           
        Returns
        -------
        dim:            dimension
        SearchSpace:         SearchSpace on original scale
        scaleSearchSpace:    SearchSpace on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.verbose=verbose
        if isinstance(SearchSpace,dict):
            # Get the name of the parameters
            self.keys = list(SearchSpace.keys())
            
            self.SearchSpace = []
            for key in list(SearchSpace.keys()):
                self.SearchSpace.append(SearchSpace[key])
            self.SearchSpace = np.asarray(self.SearchSpace)
        else:
            self.SearchSpace=np.asarray(SearchSpace)
            
            
        self.dim = len(SearchSpace)

        scaler = MinMaxScaler()
        scaler.fit(self.SearchSpace.T)
        self.Xscaler=scaler
        
        # create a scaleSearchSpace 0-1
        self.scaleSearchSpace=np.array([np.zeros(self.dim), np.ones(self.dim)]).T
                
        # function to be optimised
        self.f = func
    
        # store X in original scale
        self.X_ori= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_ori = None
        
        self.time_opt=0
         

        # acquisition function
        self.acq_name = acq_name
        self.logmarginal=0

        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=verbose)

    
    def init(self, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        
        init_X = np.random.uniform(self.SearchSpace[:, 0], self.SearchSpace[:, 1],size=(n_init_points, self.dim))
        
        self.X_ori = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_ori = np.asarray(y_init)      
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        self.X = self.Xscaler.transform(init_X)

       
        
    def init_with_data(self, init_X,init_Y,isPermutation=False):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """
          
        #init_Y=(init_Y-np.mean(init_Y))/np.std(init_Y)
            
        #outlier removal
#        idx1=np.where( init_Y<=3)[0]
#        init_Y=init_Y[idx1]
#        init_X=init_X[idx1]
#        
#        idx=np.where( init_Y>=-3)[0]
#        init_X=init_X[idx]
#        init_Y=init_Y[idx]
        
        self.Y_ori = np.asarray(init_Y)
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        
        self.X_ori=np.asarray(init_X)
        self.X = self.Xscaler.transform(init_X)
    
    def set_ls(self,lengthscale):
        self.gp.set_ls(lengthscale)
        
    def posterior(self, Xnew):
        #self.gp.fit(self.X, self.Y,IsOptimize=1)
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew)
        return mu, np.sqrt(sigma2)
        
    def select_next_point(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        #self.Y=np.reshape(self.Y,(-1,1))
        self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # optimize GP parameters after 3*dim iterations
        if  len(self.Y)%(3*self.dim)==0:
            self.gp.optimise()
            
        # Set acquisition function
        start_opt=time.time()
        x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name)

        x_max_ori=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim)))

        if self.f is None:
            return x_max,x_max_ori
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        # compute X in original scale
        
        self.X_ori=np.vstack((self.X_ori, x_max_ori))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_ori = np.append(self.Y_ori, self.f(x_max_ori))
        
        # update Y after change Y_original
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)

        return x_max#,x_max_ori
    
    
    def plot_acq_1d(self):
        x1_scale = np.linspace(self.scaleSearchSpace[0,0], self.scaleSearchSpace[0,1], 60)
        x1_scale=np.reshape(x1_scale,(-1,1))
        acq_value = self.gp_ucb(x1_scale)
        
        x1_ori=self.Xscaler.inverse_transform(x1_scale)
        
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot the surface.
        CS_acq=ax.plot(x1_ori,acq_value.reshape(x1_ori.shape))
        ax.scatter(self.X_ori[:,0],self.Y[:],marker='o',color='r',s=130,label='Obs')
      
#        temp_xaxis=np.concatenate([x1_ori, x1_ori[::-1]])
#        temp_yaxis=np.concatenate([mean_ori - 1.9600 * std, (mean_ori + 1.9600 * std)[::-1]])
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        #ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')


        ax.set_ylabel('Acquisition Function',fontsize=18)
        ax.set_xlabel('Beta',fontsize=18)
        