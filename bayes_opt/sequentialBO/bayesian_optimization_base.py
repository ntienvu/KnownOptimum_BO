# -*- coding: utf-8 -*-


import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import numpy as np

from bayes_opt.acquisition_functions import AcquisitionFunction, unique_rows
from bayes_opt.gaussian_process import GaussianProcess
from bayes_opt.acquisition_maximization import acq_max,acq_max_with_name




class BO_Sequential_Base(object):
    
    
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
        
        try:
            bounds=func_params['function']['bounds']
        except:
            bounds=func_params['function'].bounds
        
        self.dim = len(bounds)

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

 
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        # Some function to be optimized
        
        try:
            self.f = func_params['function']['func']
        except:
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
            self.optimize_gp=0
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
        
        # performance evaluation at the maximum mean GP (for information theoretic)
        self.Y_original_maxGP = None
        self.X_original_maxGP = None
        
        # value of the acquisition function at the selected point
        self.alpha_Xt=None
        self.Tau_Xt=None
        
        self.time_opt=0

        
        self.gp_params=gp_params       

        # Gaussian Process class
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
        
        # store ystars
        #self.ystars=np.empty((0,100), float)
        self.ystars=[]
        
        
    def init(self, gp_params, n_init_points=3,seed=1):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        np.random.seed(seed)
        # Generate random points
        l = [np.random.uniform(x[0], x[1]) for _ in range(n_init_points) for x in self.bounds]
        #l=[np.linspace(x[0],x[1],num=n_init_points) for x in self.init_bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        self.X_original_maxGP= np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)      
        
        self.Y_original_maxGP=np.asarray(y_init)      
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
        
        
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
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        
    def optimize_gp_hyperparameter(self,mygp=None,gp_params=None):
        
        if mygp==None:
            mygp=self.gp
            
        if gp_params==None:
            gp_params=self.gp_params
            
        if self.optimize_gp=='maximize':
            newlengthscale = mygp.optimize_lengthscale_SE_maximizing(gp_params['lengthscale'],gp_params['noise_delta'])
            gp_params['lengthscale']=newlengthscale
            #print("MML estimated lengthscale =",newlengthscale)
        elif self.optimize_gp=='loo':
            newlengthscale = mygp.optimize_lengthscale_SE_loo(gp_params['lengthscale'],gp_params['noise_delta'])
            gp_params['lengthscale']=newlengthscale
            print("LOO estimated lengthscale =",newlengthscale)

        elif self.optimize_gp=='marginal':
            self.theta_vector = mygp.slice_sampling_lengthscale_SE(gp_params['lengthscale'],gp_params['noise_delta'])
            gp_params['lengthscale']=self.theta_vector[0]
            self.theta_vector =np.unique(self.theta_vector)
            self.gp_params['newtheta_vector']=self.theta_vector 
            #print "estimated lengthscale ={:s}".format(self.theta_vector)
        elif self.optimize_gp=="fstar":
            fstar_scaled=(self.acq['fstar']-np.mean(self.Y_original))/np.std(self.Y_original)
            newlengthscale = mygp.optimize_lengthscale_SE_fstar(gp_params['lengthscale'],gp_params['noise_delta'],fstar_scaled)
            gp_params['lengthscale']=newlengthscale
            print("estimated lengthscale =",newlengthscale)
            
        tempX=mygp.X
        tempY=mygp.Y
        # init a new Gaussian Process after optimizing hyper-parameter
        mygp=GaussianProcess(gp_params)
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(tempX)
        mygp.fit(tempX[ur], tempY[ur])
        return mygp, gp_params
    
    def generate_random_point(self):
        x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
        x_max=np.asarray(x_max)
        x_max=x_max.T
        self.X_original=np.vstack((self.X_original, x_max))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(x_max))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        self.time_opt=np.hstack((self.time_opt,0))
        
        
    def augment_the_new_data(self,x_max):
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))
        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)


        # find the maximizer in the GP mean function
        try: 
            len(self.gp)
            x_mu_max=[]
            for j in range(self.J):
                x_mu_max_temp=acq_max_with_name(gp=self.gp[j],scalebounds=self.scalebounds[self.featIdx[j]],acq_name="mu")
                x_mu_max=np.hstack((x_mu_max,x_mu_max_temp))

        except:            
            x_mu_max=acq_max_with_name(gp=self.gp,scalebounds=self.scalebounds,acq_name="mu")

        
        x_mu_max_original=x_mu_max*self.max_min_gap+self.bounds[:,0]
        # set y_max = mu_max
        #mu_max=acq_mu.acq_kind(x_mu_max,gp=self.gp)
        self.Y_original_maxGP = np.append(self.Y_original_maxGP, self.f(x_mu_max_original))
        self.X_original_maxGP = np.vstack((self.X_original_maxGP, x_mu_max_original))

    