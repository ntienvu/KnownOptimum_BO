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

   
    def plot_1d(self,strTitle=None,strPath=None,starting_index=0):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        
        mean,std,mean_ori,std_ori = self.predict(x1_ori)
        
        fig = plt.figure(figsize=(6.5,5))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot the surface.
        CS_acq=ax.plot(x1_ori,mean_ori.reshape(x1_ori.shape),label="GP mean")
        #ax.scatter(self.X_ori[:,0],self.Y_ori[:],marker='o',color='r',s=130,label='Obs')
      
        temp_xaxis=np.concatenate([x1_ori, x1_ori[::-1]])
        temp_yaxis=np.concatenate([mean_ori - 1.9600 * std, (mean_ori + 1.9600 * std)[::-1]])
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        ax.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='GP var')

        colors = cm.rainbow(np.linspace(0, 1, len(self.Y_ori)))
        for ii,val in enumerate(zip(self.X_ori[:,0],self.Y_ori, colors)):
            x,y, c=val
            if ii%20==0:
                strLabel="t={:d}".format(ii*10+starting_index)
                ax.scatter(x, y, color=c,label=strLabel)
            else:
                ax.scatter(x, y, color=c)


        ax.legend(loc=2,ncol=3, fontsize=13)

        ax.set_ylabel(r'Standardized $f(\beta)$',fontsize=18)
        ax.set_xlabel(r'$\beta$',fontsize=18)
        ax.set_ylim([-3,4])
        ax.set_title(strTitle,fontsize=18)
        
        
        ax.set_xticks(np.linspace(0.02,0.74,6)) 
        todisplay=np.round(np.linspace(0,1,6).astype(float),1)
        #print(todisplay)
        ax.set_xticklabels( todisplay, fontsize=12)
        ax.set_xlim([0.02,0.74])

        print(strPath)
        fig.savefig(strPath,bbox_inches="tight")

    def plot_2d(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 60)  
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
        
        mean,std,mean_ori,std_ori = self.predict(X_ori)
        
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot the surface.
        CS_acq=ax.contourf(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape),origin='lower')
        CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower')
        ax.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=130,label='Obs')
        
        ax.set_xlabel(r'$\beta_1$',fontsize=18)
        ax.set_ylabel(r'$\beta_2$',fontsize=18)
        fig.colorbar(CS_acq, ax=ax, shrink=0.9)
        
    def plot_1d_mean_var(self):
        X_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        
        mean,std,mean_ori,std_ori = self.predict(X_ori)
        
        fig = plt.figure(figsize=(13,6))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)

        # Plot the surface.
        CS_acq=ax_mean.plot(X_ori,mean_ori.reshape(X_ori.shape))
        ax_mean.scatter(self.X_ori[:,0],self.Y_ori[:],marker='o',color='r',s=100,label='Obs')
       
        ax_mean.set_xlabel('Epoch',fontsize=18)
        ax_mean.set_ylabel('Beta',fontsize=18)
        ax_mean.set_title(r"GP Predictive Mean $\mu()$",fontsize=20)
        
        # Plot the surface.
        CS_var=ax_var.plot(X_ori,std.reshape(X_ori.shape))
        ax_var.scatter(self.X_ori[:,0],self.Y_ori,marker='o',color='r',s=100,label='Obs')
        
        
        temp_xaxis=np.concatenate([X_ori, X_ori[::-1]])
        temp_yaxis=np.concatenate([mean_ori - 1.9600 * std, (mean_ori + 1.9600 * std)[::-1]])
        #ax.scatter(self.Xdrv,Y_ori_at_drv,marker='*',s=200,color='m',label='Derivative Obs')  
        ax_var.fill(temp_xaxis, temp_yaxis,alpha=.3, fc='g', ec='None', label='95% CI')

        ax_var.set_xlabel('Epoch',fontsize=18)
        ax_var.set_ylabel('Beta',fontsize=18)
        ax_var.set_title(r"GP Predictive Var $\sigma()$",fontsize=20)
        
        
    def plot_2d_mean_var(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 50)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 50)  
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
                
        mean,std,mean_ori,std_ori = self.predict(X_ori)
        
        fig = plt.figure(figsize=(13,4.5))
        ax_mean = fig.add_subplot(1, 2, 1)
        ax_var = fig.add_subplot(1, 2, 2)

        # Plot the surface.
        CS_acq=ax_mean.contourf(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape),origin='lower')
        #CS2_acq = ax_mean.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower')
        ax_mean.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=100,label='Obs')
        
        ax_mean.set_xlim(0,1)
        ax_mean.set_ylim(0,1)

        ax_mean.set_xlabel(r'$\beta_1$',fontsize=18)
        ax_mean.set_ylabel(r'$\beta_2$',fontsize=18)
        ax_mean.set_title(r"GP Permutation Invariant $\mu( \beta)$",fontsize=20)
        fig.colorbar(CS_acq, ax=ax_mean)
        
        # Plot the surface.
        CS_var=ax_var.contourf(x1g_ori,x2g_ori,std.reshape(x1g_ori.shape),origin='lower')
        #CS2_var = ax_var.contour(CS_var, levels=CS_var.levels[::2],colors='r',origin='lower')
        ax_var.scatter(self.X_ori[:,0],self.X_ori[:,1],marker='o',color='r',s=100,label='Obs')

        ax_var.set_xlabel(r'$\beta_1$',fontsize=18)
        ax_var.set_ylabel(r'$\beta_2$',fontsize=18)
        ax_var.set_title(r"GP Permutation Invariant $\sigma( \beta)$",fontsize=20)

        ax_var.set_xlim(0,1)
        ax_var.set_ylim(0,1)

        ident = [0.0, 1.0]
        ax_var.plot(ident,ident,':',color='k')
        ax_mean.plot(ident,ident,':',color='k')

        CS_var.set_clim(0,0.9)

        fig.colorbar(CS_var, ax=ax_var)
        fig.savefig("GP2d_per_invariant.pdf",bbox_inches="tight")
        
    
    def plot_3d(self):
        x1_ori = np.linspace(self.SearchSpace[0,0], self.SearchSpace[0,1], 60)
        x2_ori = np.linspace(self.SearchSpace[1,0], self.SearchSpace[1,1], 60) 
        x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
        X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]

        #x1 = np.linspace(0, 1, 60)
        #x2 = np.linspace(0, 1, 60)    
        #x1g,x2g=np.meshgrid(x1,x2)
        #X=np.c_[x1g.flatten(), x2g.flatten()]
        
        mean,std,mean_ori,std_ori = self.predict(X_ori)
        
        fig = plt.figure(figsize=(12,7))
        ax = plt.axes(projection="3d")
        
        # Plot the surface.
        #ax.scatter(self.X_ori[:,0],self.X_ori[:,1],self.Y_ori,marker='o',color='r',s=130,label='Data')
        ax.plot_wireframe(x1g_ori,x2g_ori,mean_ori.reshape(x1g_ori.shape), color='green')

        ax.set_xlabel('Epoch',fontsize=18)
        ax.set_ylabel('Beta',fontsize=18)
        ax.set_zlabel('f(x)',fontsize=18)