# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:05:06 2020

@author: Vu Nguyen
"""

import numpy as np
from scipy.stats import norm

class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq_name):
                
        ListAcq=['bucb','ucb', 'ei', 'poi','random','thompson', 'lcb', 'mu',                     
                     'pure_exploration','kov_mes','mes','kov_ei','gp_ucb',
                         'erm','cbm','kov_tgp','kov_tgp_ei']
        # kov = know optimum value
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        #self.dim=acq['dim']
        
        
    def acq_kind(self,gp,x):
            
        y_max=np.max(gp.Y)
        
        if np.any(np.isnan(x)):
            return 0
        
        if self.acq_name == 'ucb' or self.acq_name == 'gp_ucb' :
            return self._gp_ucb( gp,x)
        if self.acq_name=='cbm':
            return self. _cbm(x,gp,target=gp.fstar)
        if self.acq_name == 'lcb':
            return self._lcb(gp,x)
        if self.acq_name == 'ei' or self.acq_name=='kov_tgp_ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'kov_ei' :
            return self._ei(x, gp, y_max=gp.fstar)
        if self.acq_name == 'erm'  or self.acq_name=='kov_ei_cb':
            return self._erm(x, gp, fstar=gp.fstar)
    #    if acq_name == 'pure_exploration':
    #        return _pure_exploration(x, gp) 
    #   
    #    if acq_name == 'mu':
    #        return _mu(x, gp)
    
    
    @staticmethod
    def _lcb(gp,xTest,fstar_scale=0):
        mean, var = gp.predict(xTest)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
#        mean=np.atleast_2d(mean).T
#        var=np.atleast_2d(var).T
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = 2 * np.log(len(gp.Y));
    
        return mean - np.sqrt(beta_t) * np.sqrt(var) 
    
#     @staticmethod
#    def _ucb(gp,xTest,fstar_scale=0):
#        dim=gp.dim
#        xTest=np.reshape(xTest,(-1,dim))
#        mean, var= gp.predict(xTest)
#        var.flags['WRITEABLE']=True
#        #var=var.copy()
#        var[var<1e-10]=0
#        mean=np.atleast_2d(mean).T
#        var=np.atleast_2d(var).T                
#        
#        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
#        beta_t =np.log(len(gp.Y))
#      
#        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
#        return mean + np.sqrt(beta_t) * np.sqrt(var) 
        
    @staticmethod
    def _gp_ucb(gp,xTest,fstar_scale=0):
        #dim=gp.dim
        #xTest=np.reshape(xTest,(-1,dim))
        mean, var= gp.predict(xTest)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        #mean=np.atleast_2d(mean).T
        #var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
      
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        temp=mean + np.sqrt(beta_t) * np.sqrt(var)
        #print("input",xTest.shape,"output",temp.shape)
        return  temp
    
    @staticmethod
    def _cbm(x, gp, target): # confidence bound minimization
        mean, var = gp.predict(x)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
#        mean=np.atleast_2d(mean).T
#        var=np.atleast_2d(var).T                
        
        # Linear in D, log in t https://github.com/kirthevasank/add-gp-bandits/blob/master/BOLibkky/getUCBUtility.m
        #beta_t = gp.X.shape[1] * np.log(len(gp.Y))
        beta_t = np.log(len(gp.Y))
      
        #beta=300*0.1*np.log(5*len(gp.Y))# delta=0.2, gamma_t=0.1
        return -np.abs(mean-target) - np.sqrt(beta_t) * np.sqrt(var) 
    
       
    @staticmethod
    def _erm(x, gp, fstar):
                
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
    
        var2 = np.maximum(var, 1e-9 + 0 * var)
        z = ( fstar-mean)/np.sqrt(var2)        
        out=(fstar-mean) * (norm.cdf(z)) + np.sqrt(var2) * norm.pdf(z)
        #print(out.shape)

        return -1*out # for minimization problem
                    
    @staticmethod
    def _ei(x, gp, y_max):
        #y_max=np.asscalar(y_max)
        mean, var = gp.predict(x)
        var2 = np.maximum(var, 1e-10 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-10]=0
        
        #print(out.shape)
        return out
       