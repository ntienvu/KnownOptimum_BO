# define Gaussian Process class


import numpy as np
from bayes_opt.acquisition_functions import unique_rows
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import scipy.linalg as spla



class GaussianProcess(object):
    
    def __init__ (self,param):
        # init the model
    
        # theta for RBF kernel exp( -theta* ||x-y||)
        if 'kernel' not in param:
            param['kernel']='SE'
            
        kernel_name=param['kernel']
        if kernel_name not in ['SE']:
            err = "The kernel function " \
                  "{} has not been implemented, " \
                  "please choose one of the kernel SE ARD.".format(kernel_name)
            raise NotImplementedError(err)
        else:
            self.kernel_name = kernel_name
            
        if 'lengthscale' not in param:
            self.lengthscale=param['theta']
        else:
            self.lengthscale=param['lengthscale']
            self.theta=self.lengthscale

        if 'lengthscale_vector' not in param: # for marginalize hyperparameters
            self.lengthscale_vector=[]
        else:
            self.lengthscale_vector=param['lengthscale_vector']
            
        #self.theta=param['theta']
        
        self.gp_params=param
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.X=[]
        self.Y=[]
        self.lengthscale_old=self.lengthscale
        self.flagOptimizeHyperFirst=0
        
        self.alpha=[] # for Cholesky update
        self.L=[] # for Cholesky update LL'=A

    def kernel_dist(self, a,b,lengthscale):
        
        if self.kernel_name=='SE':
            Euc_dist=euclidean_distances(a,b)
            return np.exp(-np.square(Euc_dist)/lengthscale)
        else:
            print("only SE kernel is implemented!")
  

        
    def fit(self,X,Y):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        y: the outcome y=f(x)
        
        """ 
        ur = unique_rows(X)
        X=X[ur]
        Y=Y[ur]
        
        self.X=X
        self.Y=Y
                
        # print("Only implemented for SE kernel")
        Euc_dist=euclidean_distances(X,X)
        self.KK_x_x=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(len(X))*self.noise_delta
       
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x")
        
        #self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        self.L=np.linalg.cholesky(self.KK_x_x)
        temp=np.linalg.solve(self.L,self.Y)
        self.alpha=np.linalg.solve(self.L.T,temp)
        
    
    def log_marginal_lengthscale(self,lengthscale,noise_delta):
        """
        Compute Log Marginal likelihood of the GP model w.r.t. the provided lengthscale
        Implementation for SE kernel
        """

        def compute_log_marginal(X,lengthscale,noise_delta):
            # compute K
            ur = unique_rows(self.X)
            myX=self.X[ur]
            myY=self.Y[ur]
            
            if self.flagOptimizeHyperFirst==0:
                self.Euc_dist_X_X=euclidean_distances(myX,myX)
                KK=np.exp(-np.square(self.Euc_dist_X_X)/lengthscale)+np.eye(len(myX))*self.noise_delta
                self.flagOptimizeHyperFirst=1
            else:
                KK=np.exp(-np.square(self.Euc_dist_X_X)/lengthscale)+np.eye(len(myX))*self.noise_delta
               
            try:
                temp_inv=np.linalg.solve(KK,myY)
            except: # singular
                return -np.inf
            
            try:
                first_term=-0.5*np.dot(myY.T,temp_inv)
                # if the matrix is too large, we randomly select a part of the data for fast computation
                if KK.shape[0]>200:
                    idx=np.random.permutation(KK.shape[0])
                    idx=idx[:200]
                    KK=KK[np.ix_(idx,idx)]
                #Wi, LW, LWi, W_logdet = pdinv(KK)
                #sign,W_logdet2=np.linalg.slogdet(KK)
                chol  = spla.cholesky(KK, lower=True)
                W_logdet=np.sum(np.log(np.diag(chol)))
                # Uses the identity that log det A = log prod diag chol A = sum log diag chol A
    
                #second_term=-0.5*W_logdet2
                second_term=-W_logdet
            except: # singular
                return -np.inf
            
            #print "first term ={:.4f} second term ={:.4f}".format(np.asscalar(first_term),np.asscalar(second_term))

            logmarginal=first_term+second_term-0.5*len(myY)*np.log(2*3.14)
                
            if np.isnan(np.asscalar(logmarginal))==True:
                print("theta={:s} first term ={:.4f} second  term ={:.4f}".format(lengthscale,np.asscalar(first_term),np.asscalar(second_term)))
                #print temp_det

            return np.asscalar(logmarginal)
        
        logmarginal=0
        
        if np.isscalar(lengthscale):
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
            return logmarginal

        if not isinstance(lengthscale,list) and len(lengthscale.shape)==2:
            logmarginal=[0]*lengthscale.shape[0]
            for idx in range(lengthscale.shape[0]):
                logmarginal[idx]=compute_log_marginal(self.X,lengthscale[idx],noise_delta)
        else:
            logmarginal=compute_log_marginal(self.X,lengthscale,noise_delta)
                
        return logmarginal
    
   
    def optimize_lengthscale_SE_maximizing(self,previous_theta,noise_delta):
        """
        Optimize to select the optimal lengthscale parameter
        """
        
        #print("maximizing lengthscale")
        dim=self.X.shape[1]
        
        # define a bound on the lengthscale
        bounds_lengthscale_min=1e-7
        bounds_lengthscale_max=0.04*dim
        mybounds=[np.asarray([bounds_lengthscale_min,bounds_lengthscale_max]).T]
        
        lengthscale_tries = np.random.uniform(bounds_lengthscale_min, bounds_lengthscale_max,size=(2*dim, 1))        
        lengthscale_tries=np.vstack((lengthscale_tries,previous_theta,bounds_lengthscale_min))
        
        # evaluate
        self.flagOptimizeHyperFirst=0 # for efficiency

        logmarginal_tries=self.log_marginal_lengthscale(lengthscale_tries,noise_delta)
        #print logmarginal_tries

        #find x optimal for init
        idx_max=np.argmax(logmarginal_tries)
        lengthscale_init_max=lengthscale_tries[idx_max]
        #print lengthscale_init_max
        
        myopts ={'maxiter':50,'maxfun':50}

        x_max=[]
        max_log_marginal=None
        
        for i in range(1):
            res = minimize(lambda x: -self.log_marginal_lengthscale(x,noise_delta),lengthscale_init_max,
                           bounds=mybounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
            if 'x' not in res:
                val=self.log_marginal_lengthscale(res,noise_delta)    
            else:
                val=self.log_marginal_lengthscale(res.x,noise_delta)  
            
            # Store it if better than previous minimum(maximum).
            if max_log_marginal is None or val >= max_log_marginal:
                if 'x' not in res:
                    x_max = res
                else:
                    x_max = res.x
                max_log_marginal = val
            #print res.x
        return x_max
    

    def optimize_lengthscale(self,previous_theta,noise_delta):
        
        if self.kernel_name=='SE':
            newlengthscale=self.optimize_lengthscale_SE_maximizing(previous_theta,noise_delta)
            self.lengthscale=newlengthscale
            
            # refit the model
            ur = unique_rows(self.X)            
            self.fit(self.X[ur],self.Y[ur])
            
            return newlengthscale

    def predict(self,xTest,eval_MSE=True):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
        # prevent singular matrix
        ur = unique_rows(self.X)
        X=self.X[ur]
           
        #             print("only SE kernel is implemented! ")

        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
        
        Euc_dist_test_train=euclidean_distances(xTest,X)
        KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)
       
        
        # using Cholesky update
        mean=np.dot(KK_xTest_xTrain,self.alpha)
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        var=KK_xTest_xTest-np.dot(v.T,v)
 
        return mean.ravel(),np.diag(var)  


    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        
  
          