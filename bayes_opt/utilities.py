# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:42:28 2020

@author: Vu Nguyen
"""

from scipy.optimize import minimize
import numpy as np
import time
from bayes_opt.acq_functions import AcquisitionFunction
import sys
import pickle
import os

out_dir="pickle_storage"


def acq_max_with_name(gp,SearchSpace,acq_name="ei",IsReturnY=False,IsMax=True,fstar_scaled=None):
 
    acq=AcquisitionFunction(acq_name)
    if IsMax:
        x_max = acq_max_scipy(gp,acq.acq_kind,SearchSpace)
    else:
        x_max = acq_min_scipy(gp,acq.acq_kind,SearchSpace)
    if IsReturnY==True:
        y_max=acq.acq_kind(gp,x_max)
        return x_max,y_max
    return x_max


def acq_max_scipy( gp,acq, SearchSpace):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = SearchSpace[:, 0]
    max_acq = None

    dim=gp.dim
    myopts ={'maxiter':50*dim,'maxfun':50*dim}

    for i in range(3*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(10*dim, dim))
    
        # evaluate
        y_tries=acq(gp,x_tries)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)

        x_init_max=x_tries[idx_max]
    
        res = minimize(lambda x: -acq( gp, x.reshape(1, -1)),x_init_max.reshape(1, -1),bounds=SearchSpace,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B
 
        if 'x' not in res:
            val=acq(gp,res)        
        else:
            val=acq(gp,res.x) 
                
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val

    return np.clip(x_max, SearchSpace[:, 0], SearchSpace[:, 1])
    

def acq_min_scipy(gp,acq,SearchSpace):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    SearchSpace: The variables SearchSpace to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = SearchSpace[:, 0]
    min_acq = None

    dim=gp.dim
    myopts ={'maxiter':50*dim,'maxfun':50*dim}

    # multi start
    for i in range(3*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(10*dim, dim))
    
        # evaluate
        y_tries=acq(gp,x_tries)
        
        #find x optimal for init
        x_init_max=x_tries[np.argmin(y_tries)]
        
        res = minimize(lambda x: acq(gp,x.reshape(1, -1)),x_init_max.reshape(1, -1),
               bounds=SearchSpace,method="L-BFGS-B",options=myopts)#L-BFGS-B
  
        val=acq(gp,res.x) 

        # Store it if better than previous minimum(maximum).
        if min_acq is None or val <= min_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            min_acq = val


    return np.clip(x_max, SearchSpace[:, 0], SearchSpace[:, 1])

def run_experiment(bo,yoptimal=0,n_init=3,NN=10,runid=1):
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(n_init_points=n_init,seed=runid)
    
    # number of recommended parameters
    for idx in range(0,NN):
        bo.select_next_point()

    fxoptimal=bo.Y_ori
    elapsed_time = time.time() - start_time

    return fxoptimal, elapsed_time

    
def yBest_Iteration(YY,BatchSzArray,step=3):
    
    nRepeat=len(YY)
    
    result=[0]*nRepeat

    for ii,yy in enumerate(YY):
        result[ii]=[np.max(yy[:uu+1]) for uu in range(len(yy))]
        
    result=np.asarray(result)
    
    result_mean=np.mean(result,axis=0)
    result_mean=result_mean[BatchSzArray[0]-1:]
    result_std=np.std(result,axis=0)
    result_std=result_std[BatchSzArray[0]-1:]
    
    return result_mean[::step], result_std[::step], None, None

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



def print_result_sequential(bo,myfunction,Score,acq_type):
    
    if 'ystars' in acq_type:
        acq_type['ystars']=[]
    if 'xstars' in acq_type:
        acq_type['xstars']=[]
        
    #Regret=Score["Regret"]
    ybest=Score["ybest"]
    #GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    
    print('{:s} {:d}'.format(myfunction.name,myfunction.input_dim))
    print(acq_type['name'],acq_type['IsTGP'])
    
 
    MaxFx=[val.max() for idx,val in enumerate(ybest)]

    
  
    if myfunction.ismax==1:
        print('MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))    
    else:
        print('MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))
            
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]
        print('OptTime/Iter={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime)))
        
    if acq_type['IsTGP']==1: # using Transformed GP
        strFile="{:s}_{:d}_{:s}_TGP.pickle".format(myfunction.name,myfunction.input_dim,acq_type['name'])
    else: # using GP
        strFile="{:s}_{:d}_{:s}_GP.pickle".format(myfunction.name,myfunction.input_dim,acq_type['name'])
    
    if sys.version_info[0] < 3:
        version=2
    else:
        version=3
        
    path=os.path.join(out_dir,strFile)
    
    if version==2:
        with open(path, 'wb') as f:
            pickle.dump([ybest, MyTime,bo[-1].bounds,MyOptTime], f)
    else:
        pickle.dump( [ybest, MyTime,bo,MyOptTime], open( path, "wb" ) )
