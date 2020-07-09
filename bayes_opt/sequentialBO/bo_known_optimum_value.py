# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2018

"""


import numpy as np
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.gaussian_process import GaussianProcess
from bayes_opt.transform_gp import TransformedGP
from sklearn.cluster import KMeans
from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name
from bayes_opt.utility.basic_utility_functions import generate_random_points
import time

#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt_KnownOptimumValue(object):

    def __init__(self, gp_params, func_params, acq_params,verbose=1):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
        
        
        isTGP: using transformed Gaussian process
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        if verbose==1:
            self.verbose=1
        else:
            self.verbose=0
        # Find number of parameters
        bounds=func_params['function'].bounds
        if 'init_bounds' not in func_params:
            init_bounds=bounds
        else:
            init_bounds=func_params['init_bounds']
        
        self.dim = len(bounds)

        self.fstar=func_params['function'].fstar
        
        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in list(bounds.keys()):
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)

        if len(init_bounds)==0:
            self.init_bounds=self.bounds.copy()
        else:
            self.init_bounds=init_bounds
            
        if isinstance(init_bounds,dict):
            # Get the name of the parameters
            self.keys = list(init_bounds.keys())
        
            self.init_bounds = []
            for key in list(init_bounds.keys()):
                self.init_bounds.append(init_bounds[key])
            self.init_bounds = np.asarray(self.init_bounds)
        else:
            self.init_bounds=np.asarray(init_bounds)            
            
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        
        # Some function to be optimized
        self.f = func_params['function'].func
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        # acquisition function type
        
        self.acq=acq_params['acq_func']
        self.acq['scalebounds']=self.scalebounds
        
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp='maximize'
        else:                
            self.optimize_gp=acq_params['optimize_gp']       
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
            
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        

        self.time_opt=0

        self.gp_params=gp_params       

        # Gaussian Process class
        if 'surrogate' not in self.acq:
            self.acq['surrogate']='gp'

        if self.acq['surrogate']=='tgp':
            self.isTGP=1
            self.gp=TransformedGP(gp_params)
        else:
            self.isTGP=0
            self.gp=GaussianProcess(gp_params)

            
        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.xstar_accumulate=[]

        # theta vector for marginalization GP
        self.theta_vector =[]
        
        if 'n_xstars' in self.acq:
            self.numXstar=self.acq['n_xstars']
        else:
            #self.numXstar=self.dim*50
            self.numXstar=100
        # store ystars
        #self.ystars=np.empty((0,100), float)
        self.gstars=np.empty((0,self.numXstar), float)
        
        
        self.gap_gstar_fstar=np.empty((0,self.numXstar), float)
        
        # store all selection of AF for algorithm with confidence bound
        self.marker=[]
        
        self.flagTheta_TS=0
        self.mean_theta_TS=None
        # will be later used for visualization
        
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    def posterior_tgp(self, Xnew):
        fstar_scaled=(self.fstar-np.mean(self.Y_original))/np.std(self.Y_original)

        self.gp.fit(self.X, self.Y,fstar_scaled)
        
        x_ucb,y_ucb=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="ucb",IsReturnY=True)
        x_lcb,y_lcb=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="lcb",IsReturnY=True,IsMax=False)
        
        #print("y_lcb={} y_ucb={} fstar_scaled={:4f}".format(y_lcb,y_ucb,fstar_scaled))
        if y_lcb>fstar_scaled or y_ucb<fstar_scaled: # f* > ucb
            self.gp.isZeroMean=True
            self.gp_params['isZeroMean']=True
            print("ZeroMean")
            self.gp.fit(self.X, self.Y,fstar_scaled)
        else:
            self.gp.isZeroMean=False
            self.gp_params['isZeroMean']=False
            
            
            
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    def posterior_tgp_g(self, Xnew):
        fstar_scaled=(self.fstar-np.mean(self.Y_original))/np.std(self.Y_original)

        self.tgp.fit(self.X, self.Y,fstar_scaled)
        mu, sigma2 = self.tgp.predict_G(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
        

    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        
        self.Y_original = np.asarray(init_Y)
        #self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)        
        
        # add y_optimum into Y set
        #Y_temp=[self.Y_original,self.fstar]
        #self.Y=(self.Y_original-np.mean(Y_temp))/np.std(Y_temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)


        
    def init(self, gp_params, n_init_points=3, seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        # Generate random points
        l = [np.random.uniform(x[0], x[1]) for _ in range(n_init_points) for x in self.init_bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)        
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
   
        
    def maximize(self):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            x_max=generate_random_points(bounds=self.scalebounds,size=1)
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return

        fstar_scaled=(self.fstar-np.mean(self.Y_original))/np.std(self.Y_original)
        self.acq['fstar_scaled']=np.asarray([fstar_scaled])
            
        # init a new Gaussian Process
        if self.isTGP==1:
            self.gp=TransformedGP(self.gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
        else:
            self.gp=GaussianProcess(self.gp_params)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
            
             
        # check if the surrogate hit the optimum value f*, check if UCB and LCB cover the fstar
        x_ucb,y_ucb=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="ucb",IsReturnY=True,fstar_scaled=fstar_scaled)
        x_lcb,y_lcb=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="lcb",IsReturnY=True,IsMax=False)
        
        if y_lcb>fstar_scaled or y_ucb<fstar_scaled: # f* > ucb
            self.gp.isZeroMean=True
            self.gp_params['isZeroMean']=True
            print("y_lcb={} y_ucb={} fstar_scaled={:4f}".format(y_lcb,y_ucb,fstar_scaled))
            print("ZeroMean")
        else:
            self.gp.isZeroMean=False
            self.gp_params['isZeroMean']=False


        # optimize GP parameters after 10 iterations
        newlengthscale=None
        # we donot optimize lengthscale for the setting of gp_lengthscale
        if  len(self.Y)%(5*self.dim)==0:
            if self.optimize_gp=='maximize':
                newlengthscale = self.gp.optimize_lengthscale_SE_maximizing(self.gp_params['lengthscale'],self.gp_params['noise_delta'])
                self.gp_params['lengthscale']=newlengthscale
                
            elif self.optimize_gp=='loo':
                newlengthscale = self.gp.optimize_lengthscale_SE_loo(self.gp_params['lengthscale'],self.gp_params['noise_delta'])
                self.gp_params['lengthscale']=newlengthscale

         
            if self.verbose==1:
                print("estimated lengthscale =",newlengthscale)

            # init a new Gaussian Process after optimizing hyper-parameter
            if self.isTGP==1:
                self.gp=TransformedGP(self.gp_params)
                # Find unique rows of X to avoid GP from breaking
                ur = unique_rows(self.X)
                self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
            else:
                self.gp=GaussianProcess(self.gp_params)
                ur = unique_rows(self.X)
                self.gp.fit(self.X[ur], self.Y[ur])

 
        # Set acquisition function
        start_opt=time.time()
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        
        x_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name=self.acq['name'],fstar_scaled=fstar_scaled)


        if np.any(np.abs((self.X - x_max)).sum(axis=1) <= (self.dim*1e-4)):
            print("{} x_max is repeated".format(self.acq['name']))
            
            self.gp.isZeroMean=True
            self.gp_params['isZeroMean']=True
            self.gp=TransformedGP(self.gp_params)
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur],fstar_scaled)
            
            x_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name=self.acq['name'],fstar_scaled=fstar_scaled)
            #self.gp_params['lengthscale']=self.gp_params['lengthscale']-0.01
            #x_max=generate_random_points(self.scalebounds,1)
            
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
   
                              
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        y_original=self.f(temp_X_new_original)
        self.Y_original = np.append(self.Y_original, y_original)
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
    