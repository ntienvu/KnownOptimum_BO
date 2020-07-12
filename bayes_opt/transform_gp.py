
# define Gaussian Process class

import numpy as np
from scipy.optimize import minimize
#from bayes_opt.acquisition_functions import unique_rows
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
#import scipy.linalg as spla
from bayes_opt.utilities import unique_rows
import scipy

class TransformedGP(object):
    # transform GP given known optimum value: f = f^* - 1/2 g^2
    def __init__ (self,SearchSpace,fstar=None,noise_delta=1e-8,verbose=0,IsZeroMean=False):
        # init the model
    
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
        self.hyper['lengthscale']=0.035 #to be optimised
        #self.hyper['noise_delta']=noise_delta # could be optimised
        self.noise_delta=self.noise_delta
        self.fstar=fstar
        self.IsZeroMean=IsZeroMean

        
#        self.KK_x_x=[]
#        self.KK_x_x_inv=[]
#    
#        self.fstar=0
#        self.X=[]
#        self.Y=[]
#        self.G=[]
#        self.lengthscale_old=self.lengthscale
#        self.flagOptimizeHyperFirst=0
#        
#        self.alpha=[] # for Cholesky update
#        self.L=[] # for Cholesky update LL'=A
#    def set_optimum_value(self,fstar_scaled):
#        self.fstar_scaled=fstar_scaled

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
        
    
        
    def fit(self,X,Y,fstar=None,IsOptimize=0):
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
        if fstar is not None:
            self.fstar=fstar
        self.G=np.sqrt(2.0*(fstar-Y))
        #self.G=np.log(1.0*(fstar-Y))
        
        # print("only SE kernel is implemented!")
        #Euc_dist=euclidean_distances(X,X)
        
        if IsOptimize:
            self.hyper['lengthscale']=self.optimise()         # optimise GP hyperparameters
        #self.hyper['epsilon'],self.hyper['lengthscale'],self.noise_delta=self.optimise()         # optimise GP hyperparameters
        self.KK_x_x=self.mycov(self.X,self.X,self.hyper)+np.eye(len(X))*self.noise_delta 
        #self.KK_x_x=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(len(X))*self.noise_delta
        
        if np.isnan(self.KK_x_x).any(): #NaN
            print("nan in KK_x_x")
        
   
        self.L=np.linalg.cholesky(self.KK_x_x)
        
        # no zero mean
        
        # zero mean
        if self.IsZeroMean:
            tempG=np.linalg.solve(self.L,self.G)
        else:
            tempG=np.linalg.solve(self.L,self.G-np.sqrt(2*self.fstar))
        
        #self.alpha=np.linalg.solve(self.L.T,temp)
        self.alphaG=np.linalg.solve(self.L.T,tempG)
        
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
        bounds=np.asarray([[1e-2,1]])

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


  
    def predict_g2(self,xTest,eval_MSE=True):
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
        #Y=self.Y[ur]
        #G=self.G[ur]
            
        # print("only SE kernel is implemented!")
        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
        
        Euc_dist_test_train=euclidean_distances(xTest,X)
        KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)        
        
        # using Cholesky update
        meanG=np.dot(KK_xTest_xTrain,self.alphaG)
        
        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        
        
        # compute mF, varF
        mf=self.fstar-0.5*meanG*meanG
        varf=meanG*varG*meanG
        #varf=varG

        return mf.ravel(),np.diag(varf)     
    
    def predict(self,Xtest,isOriScale=False):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if isOriScale:
            Xtest=self.Xscaler.transform(Xtest)
            
        if len(Xtest.shape)==1: # 1d
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        if Xtest.shape[1] != self.X.shape[1]: # different dimension
            Xtest=np.reshape(Xtest,(-1,self.X.shape[1]))
            
        KK_xTest_xTest=self.mycov(Xtest,Xtest,self.hyper)+np.eye(Xtest.shape[0])*self.noise_delta
        KK_xTest_xTrain=self.mycov(Xtest,self.X,self.hyper)
        
       
        # using Cholesky update
        
        if self.IsZeroMean:
            meanG=np.dot(KK_xTest_xTrain,self.alphaG) # zero prior mean
        else:
            meanG=np.dot(KK_xTest_xTrain,self.alphaG)+np.sqrt(2*self.fstar) # non zero prior mean
        

        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        
        # compute mF, varF
        mf=self.fstar-0.5*np.square(meanG)
        #mf=self.fstar-np.exp(meanG)
        
        # using linearlisation
        varf=meanG*varG*meanG 
        # using moment matching
        
    
        return np.reshape(mf.ravel(),(-1,1)),np.reshape(np.diag(varf)  ,(-1,1))

    def predict_G(self,xTest,eval_MSE=True):
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
        #Y=self.Y[ur]
        #G=self.G[ur]
    
        
        #print("only SE kernel is implemented")
        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-np.square(Euc_dist)/self.lengthscale)+np.eye(xTest.shape[0])*self.noise_delta
        
        Euc_dist_test_train=euclidean_distances(xTest,X)
        KK_xTest_xTrain=np.exp(-np.square(Euc_dist_test_train)/self.lengthscale)

        
        meanG=np.dot(KK_xTest_xTrain,self.alphaG)+np.sqrt(2*self.fstar) # non zero prior mean

        v=np.linalg.solve(self.L,KK_xTest_xTrain.T)
        varG=KK_xTest_xTest-np.dot(v.T,v)
        
        return meanG.ravel(),np.diag(varG)  

    
    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        
    
