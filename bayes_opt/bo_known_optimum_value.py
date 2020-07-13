# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2018

"""


import numpy as np
#from scipy.optimize import minimize
#from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.transform_gp import TransformedGP
from bayes_opt.gp import GaussianProcess
#from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name
#from bayes_opt.utilities import acq_max_scipy
import time
from sklearn.preprocessing import MinMaxScaler
from bayes_opt.utilities import unique_rows
from bayes_opt.utilities import acq_max_with_name

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt_KnownOptimumValue(object):

    #def __init__(self, gp_params,SearchSpace, acq_func,verbose=1):
    def __init__(self, func, SearchSpace,fstar=0,acq_name="erm",IsTGP=1,verbose=1):
        """      
        Input parameters
        ----------
        
        func:                       a function to be optimized
        SearchSpace:                bounds on parameters        
        fstar:                      known optimum value of the black-box function
                
        acq_func:            acquisition function ['erm','cbm']
        # erm: expected regret minimization
        # cbm: confidence bound minimization
       
        isTGP: ={0,1} either using vanilla GP or transformed Gaussian process
        verbose = {0,1}: printing the intermediate result
        
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scaleSearchSpace:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        if verbose==1:
            self.verbose=1
        else:
            self.verbose=0
        # Find number of parameters
        
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

        self.fstar=fstar
    
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scaleSearchSpace=scalebounds.T
        
        scaler = MinMaxScaler()
        scaler.fit(self.SearchSpace.T)
        self.Xscaler=scaler
        
        # Some function to be optimized
        self.f = func
        self.acq_name = acq_name

      
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

        self.IsZeroMean=False # this will be automatically changed

        # Gaussian Process class
        self.IsTGP=IsTGP
        if self.IsTGP==1:
            self.gp=TransformedGP(self.scaleSearchSpace,verbose=verbose,IsZeroMean=self.IsZeroMean)
        else:
            self.gp=GaussianProcess(self.scaleSearchSpace,verbose=verbose)

            
        # acquisition function
        self.acq_func = None
    
        self.logmarginal=0

        # store all selection of AF for algorithm with confidence bound
        self.marker=[]
        
        self.flagTheta_TS=0
        self.mean_theta_TS=None
        # will be later used for visualization
        
        self.marker=[]
        
    def set_ls(self,lengthscale):
        self.gp.set_ls(lengthscale)
        
    def posterior(self, Xnew):
        #self.gp.fit(self.X, self.Y,IsOptimize=1)
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew)
        return mu, np.sqrt(sigma2)
    
    def posterior_tgp(self, Xnew):
        fstar_scaled=(self.fstar-np.mean(self.Y_ori))/np.std(self.Y_ori)

        #self.gp.fit(self.X, self.Y,fstar_scaled,IsOptimize=1)
        self.gp.fit(self.X, self.Y,fstar_scaled)

        x_ucb,y_ucb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="ucb",IsReturnY=True)
        x_lcb,y_lcb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="lcb",IsReturnY=True,IsMax=False)
        
        print("y_lcb={} y_ucb={} fstar_scaled={:4f}".format(y_lcb,y_ucb,fstar_scaled))
        if y_lcb>fstar_scaled or y_ucb<fstar_scaled: # f* > ucb
            self.gp.IsZeroMean=True
            self.IsZeroMean=True
            print("ZeroMean")
            self.gp.fit(self.X, self.Y,fstar_scaled)
        else:
            self.gp.IsZeroMean=False
            self.IsZeroMean=False
            

        #self.gp.optimise()
            
        mu, sigma2 = self.gp.predict(Xnew)
        return mu, np.sqrt(sigma2)
    
#    def posterior_tgp_g(self, Xnew):
#        fstar_scaled=(self.fstar-np.mean(self.Y_ori))/np.std(self.Y_ori)
#
#        self.tgp.fit(self.X, self.Y,fstar_scaled)
#        mu, sigma2 = self.tgp.predict_G(Xnew)
#        return mu, np.sqrt(sigma2)
        

    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        # Turn it into np array and store.
        self.X_ori=np.asarray(init_X)
        
        self.X_ori = np.asarray(init_X)
        self.X = self.Xscaler.transform(init_X)
        
        self.Y_ori = np.asarray(init_Y)
        
        # add y_optimum into Y set
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)


        
    def init(self, n_init_points=3, seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        # Generate random points
        init_X = np.random.uniform(self.SearchSpace[:, 0], self.SearchSpace[:, 1],size=(n_init_points, self.dim))
   
        self.X_ori = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_ori = np.asarray(y_init)        
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)

        self.X = self.Xscaler.transform(init_X)

    def select_next_point(self):
        """
        select the next point to evaluate
        -------
        x: recommented point for evaluation
        """

        fstar_scaled=(self.fstar-np.mean(self.Y_ori))/np.std(self.Y_ori)
            
        # init a new Gaussian Process
        if self.IsTGP==1:
            self.gp=TransformedGP(self.scaleSearchSpace,verbose=self.verbose,IsZeroMean=self.IsZeroMean)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
        else:
            self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            self.gp.set_optimum_value(fstar_scaled)
            
        # optimize GP parameters after 3*dim iterations
        if  len(self.Y)%(3*self.dim)==0:
            self.gp.optimise()
            
        # check if the surrogate hit the optimum value f*, check if UCB and LCB cover the fstar
        x_ucb,y_ucb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="ucb",IsReturnY=True,fstar_scaled=fstar_scaled)
        x_lcb,y_lcb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="lcb",IsReturnY=True,IsMax=False)
        
        #start_opt=time.time()

        if y_ucb<fstar_scaled: # f* > ucb we initially use EI with the vanilla GP until the fstar is covered by the upper bound
            x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="ei")
            self.marker.append(0)
            if self.verbose==1:
                print("y_lcb={} y_ucb={} fstar_scaled={:.4f}".format(y_lcb,y_ucb,fstar_scaled))
                print("EI")
        else: # perform TransformGP and ERM/CBM
            self.marker.append(1)
            self.IsTGP=1
            x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name, \
                                    fstar_scaled=fstar_scaled)

        if np.any(np.abs((self.X - x_max)).sum(axis=1) <= (self.dim*1e-5)):
            
            if self.verbose==1:
                print("{} x_max is repeated".format(self.acq_name))
                
            # lift up the surrogate function
            self.gp.IsZeroMean=True
            self.IsZeroMean=True
            self.gp=TransformedGP(self.scaleSearchSpace,verbose=self.verbose,IsZeroMean=self.IsZeroMean)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
            
            x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name,fstar_scaled=fstar_scaled)
            
   
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        x_max_ori=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim)))

        self.X_ori=np.vstack((self.X_ori, x_max_ori))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        Y_ori=self.f(x_max_ori)
        self.Y_ori = np.append(self.Y_ori, Y_ori)
        
        # update Y after change Y_ori
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        
        return x_max   
        
    def select_next_point_bk(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        fstar_scaled=(self.fstar-np.mean(self.Y_ori))/np.std(self.Y_ori)
            
        # init a new Gaussian Process
        if self.IsTGP==1:
            self.gp=TransformedGP(self.scaleSearchSpace,verbose=self.verbose,IsZeroMean=self.IsZeroMean)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
        else:
            self.gp=GaussianProcess(self.scaleSearchSpace,verbose=self.verbose)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            

        # check if the surrogate hit the optimum value f*, check if UCB and LCB cover the fstar
        x_ucb,y_ucb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="ucb",IsReturnY=True,fstar_scaled=fstar_scaled)
        x_lcb,y_lcb=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name="lcb",IsReturnY=True,IsMax=False)
        
        if y_lcb>fstar_scaled or y_ucb<fstar_scaled: # f* > ucb
            self.gp.IsZeroMean=True
            self.IsZeroMean=True

            if self.verbose==1:
                print("y_lcb={} y_ucb={} fstar_scaled={:.4f}".format(y_lcb,y_ucb,fstar_scaled))
                print("ZeroMean")
        else:
            self.gp.IsZeroMean=False
            self.IsZeroMean=False

        # optimize GP parameters after 3*dim iterations
        if  len(self.Y)%(3*self.dim)==0:
            self.gp.optimise()
 
        # Set acquisition function
        start_opt=time.time()
        # run the acquisition function for the first time to get xstar
                
        x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name,fstar_scaled=fstar_scaled)

        if np.any(np.abs((self.X - x_max)).sum(axis=1) <= (self.dim*1e-5)):
            
            if self.verbose==1:
                print("{} x_max is repeated".format(self.acq_name))
                
            # lift up the surrogate function
            self.gp.IsZeroMean=True
            self.IsZeroMean=True
            self.gp=TransformedGP(self.scaleSearchSpace,verbose=self.verbose,IsZeroMean=self.IsZeroMean)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
            
            x_max=acq_max_with_name(gp=self.gp,SearchSpace=self.scaleSearchSpace,acq_name=self.acq_name,fstar_scaled=fstar_scaled)
            
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
                              
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        x_max_ori=self.Xscaler.inverse_transform(np.reshape(x_max,(-1,self.dim)))

        self.X_ori=np.vstack((self.X_ori, x_max_ori))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        Y_ori=self.f(x_max_ori)
        self.Y_ori = np.append(self.Y_ori, Y_ori)
        
        # update Y after change Y_ori
        self.Y=(self.Y_ori-np.mean(self.Y_ori))/np.std(self.Y_ori)
        
        return x_max