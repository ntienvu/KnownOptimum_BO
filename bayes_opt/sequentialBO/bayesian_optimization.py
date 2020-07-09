# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


import numpy as np
#from sklearn.gaussian_process import GaussianProcess

from bayes_opt.sequentialBO.bayesian_optimization_base import BO_Sequential_Base
from scipy.optimize import minimize
from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.gaussian_process import GaussianProcess

from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name


import time

#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class BayesOpt(BO_Sequential_Base):

    def __init__(self, gp_params, func_params, acq_params, verbose=1):
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
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        # Find number of parameters
        
        super(BayesOpt, self).__init__(gp_params, func_params, acq_params, verbose)

       
       
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, gp_params, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        super(BayesOpt, self).init(gp_params, n_init_points,seed)
        
    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """

        super(BayesOpt, self).init_with_data(init_X,init_Y)
        
           
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        

            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]

        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})        
   
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
                
        L=-minusL
        if L<1e-6: L=0.0001  ## to avoid problems in cases in which the model is flat.
        
        return L    
      
            
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
            
            super(BayesOpt, self).generate_random_point()

            return

        # init a new Gaussian Process
        self.gp=GaussianProcess(self.gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq
       
        # optimize GP parameters after 10 iterations
        if  len(self.Y)%(5*self.dim)==0:
            self.gp,self.gp_params=super(BayesOpt, self).optimize_gp_hyperparameter()



        if self.acq['name']=='mes':
            self.maximize_mes()
            return

        # Set acquisition function
        start_opt=time.time()

        #y_max = self.Y.max()
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean            
            x_mu_max,y_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name='mu',IsReturnY=True)
 
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)


        val_acq=self.acq_func.acq_kind(x_max,self.gp)

        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            #val_acq=self.acq_func.acq_kind(x_max,self.gp)

            self.stop_flag=1
            #print "Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria)
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        
        super(BayesOpt, self).augment_the_new_data(x_max)

        
  
    def maximize_mes(self):
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
               
 
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                
            
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        # finding the xt of UCB
        
        y_max=np.max(self.Y)
        #numXtar=10*self.dim
        numXtar=30
        
        y_stars=[]
        temp=[]
        # finding the xt of Thompson Sampling
        for ii in range(numXtar):
            
            xt_TS,y_xt_TS=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,
                                                acq_name="thompson",IsReturnY=True)
            
            #if y_xt_TS>=y_max:
            y_stars.append(y_xt_TS)
            
            temp.append(xt_TS)
            # check if f* > y^max and ignore xt_TS otherwise
            if y_xt_TS>=y_max:
                self.xstars.append(xt_TS)

        if self.xstars==[]:
            #print 'xt_suggestion is empty'
            # again perform TS and take all of them
            self.xstars=temp
            
        self.acq['xstars']=self.xstars   
        self.acq['ystars']=y_stars   


        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)

        #xstars_array=np.asarray(self.acq_func.object.xstars)

        val_acq=self.acq_func.acq_kind(x_max,self.gp)
        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        super(BayesOpt, self).augment_the_new_data(x_max)

        
        # convert ystar to original scale
        y_stars=[val*np.std(self.Y_original)+np.mean(self.Y_original) for idx,val in enumerate(y_stars)]
        
        self.ystars.append(np.ravel(y_stars))
 

#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
