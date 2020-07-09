# -*- coding: utf-8 -*-


import sys
sys.path.insert(0,'../..')
sys.path.insert(0,'../')
#from prada_bayes_opt import PradaBayOptFn

#from sklearn.gaussian_process import GaussianProcess
#from scipy.stats import norm
#import matplotlib as plt
#from prada_bayes_opt.batchBO import bayesian_optimization_batch
#from prada_bayes_opt import bayesian_optimization_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from bayes_opt import PradaBayesianOptimization
import numpy as np

import pickle
import os
import sys

#out_dir="P:\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
out_dir="pickle_storage"

def print_result_sequential(bo,myfunction,Score,acq_type,toolbox='PradaBO'):
    
    if 'ystars' in acq_type:
        acq_type['ystars']=[]
    if 'xstars' in acq_type:
        acq_type['xstars']=[]
        
    #Regret=Score["Regret"]
    ybest=Score["ybest"]
    #GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    
    print('{:s} {:d}'.format(myfunction.name,myfunction.input_dim))
    print(acq_type['name'],acq_type['surrogate'])
    
    if toolbox=='GPyOpt':
        MaxFx=[val.min() for idx,val in enumerate(ybest)]
    else:
        MaxFx=[val.max() for idx,val in enumerate(ybest)]

    
    if toolbox=='GPyOpt':
        if myfunction.ismax==1:
            print('MaxBest={:.4f}({:.2f})'.format(-1*np.mean(MaxFx),np.std(MaxFx)))    
        else:
            print('MinBest={:.4f}({:.2f})'.format(np.mean(MaxFx),np.std(MaxFx)))
    else:            
        if myfunction.ismax==1:
            print('MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))    
        else:
            print('MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx)))
            
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]
        if toolbox=='GPyOpt':
            print('OptTime/Iter={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime)))
        else:
            print('OptTime/Iter={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime)))
        
    strFile="{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,acq_type['name'],acq_type['surrogate'])
    
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

  
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0,step=3):
    
    nRepeat=len(YY)
    YY=np.asarray(YY)
    ##YY_mean=np.mean(YY,axis=0)
    #YY_std=np.std(YY,axis=0)
    
    mean_TT=[]
    #temp_std=np.std(YY[:,0:BatchSzArray[0]+1])
    #temp_std=np.std(YY_mean[0:BatchSzArray[0]+1])
    
    mean_cum_TT=[]
    
    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
    
        if IsPradaBO==1:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].max()
        else:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].min()
        
        temp_mean_cum=YY[idxtt,0:BatchSzArray[0]+1].mean()

        start_point=0
        for idx,bz in enumerate(BatchSzArray): # batch
            if idx==len(BatchSzArray)-1:
                break
            bz=np.int(bz)

            #    get the average in this batch
            temp_mean_cum=np.vstack((temp_mean_cum,YY[idxtt,start_point:start_point+bz].mean()))
            
            # find maximum in each batch            
            if IsPradaBO==1:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].max()))
            else:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].min()))

            start_point=start_point+bz

        if IsPradaBO==1:
            myYbest=[temp_mean[:idx+1].max()*-1 for idx,val in enumerate(temp_mean)]
            temp_mean_cum=temp_mean_cum*-1
            temp_mean=temp_mean*-1
        else:
            myYbest=[temp_mean[:idx+1].min() for idx,val in enumerate(temp_mean)]
        
        # cumulative regret for each independent run
        #myYbest_cum=[np.mean(np.abs(temp_mean_cum[:idx+1]-Y_optimal)) for idx,val in enumerate(temp_mean_cum)]
        
        temp_regret=np.abs(temp_mean-Y_optimal)
        myYbest_cum=[np.mean(temp_regret[:idx+1]) for idx,val in enumerate(temp_regret)]


        if len(mean_TT)==0:
            mean_TT=myYbest
            mean_cum_TT=myYbest_cum
        else:
            #mean_TT.append(temp_mean)
            mean_TT=np.vstack((mean_TT,myYbest))
            mean_cum_TT=np.vstack((mean_cum_TT,myYbest_cum))
            
    mean_TT    =np.array(mean_TT)
    std_TT=np.std(mean_TT,axis=0)
    std_TT=np.array(std_TT).ravel()
    mean_TT=np.mean(mean_TT,axis=0)

    
    mean_cum_TT=np.array(mean_cum_TT)   
    std_cum_TT=np.std(mean_cum_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_TT=np.mean(mean_cum_TT,axis=0)
   
    #return mean_TT[::step],std_TT[::step]#,mean_cum_TT[::5],std_cum_TT[::5]
    return mean_TT[::step],std_TT[::step],mean_cum_TT[::step],std_cum_TT[::step]


def compute_average_cumulative_simple_regret(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0):
        
    nRepeat=len(YY)
    YY=np.asarray(YY)
    
    #half_list_index=np.int(len(YY[0])*0.5)
    half_list_index=BatchSzArray[0]+1
    #half_list_index=1

    # remove first half
   # mean_TT=[]

    mean_cum_simple_regret_TT=[]
    
    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
    
        if IsPradaBO==1:
            temp_simple_regret=YY[idxtt,0:BatchSzArray[0]+1].max()
        else:
            temp_simple_regret=YY[idxtt,0:BatchSzArray[0]+1].min()
        

        start_point=0
        for idx,bz in enumerate(BatchSzArray): # batch
            if idx==0:
                continue
            if idx==len(BatchSzArray)-1:
                break
            bz=np.int(bz)
            
            # find maximum in each batch            
            if IsPradaBO==1:
                temp_simple_regret=np.vstack((temp_simple_regret,YY[idxtt,start_point:start_point+bz].max()))
            else:
                temp_simple_regret=np.vstack((temp_simple_regret,YY[idxtt,start_point:start_point+bz].min()))

            start_point=start_point+bz

        if IsPradaBO==1:
            # ignore the first element of initialization
            myYbest=[temp_simple_regret[:idx+1].max()*-1 for idx,val in enumerate(temp_simple_regret)]
            temp_simple_regret=temp_simple_regret*-1
        else:
            myYbest=[temp_simple_regret[:idx+1].min() for idx,val in enumerate(temp_simple_regret)]
        
        # cumulative regret for each independent run
        #myYbest_cum=[np.mean(np.abs(temp_mean_cum[:idx+1]-Y_optimal)) for idx,val in enumerate(temp_mean_cum)]
        
        temp_regret=np.abs(np.asarray(myYbest)-Y_optimal)
        temp_regret=temp_regret[half_list_index:]
        myYbest_cum=[np.mean(temp_regret[:idx+1]) for idx,val in enumerate(temp_regret)]

        
        
        if len(mean_cum_simple_regret_TT)==0:
            #mean_TT=myYbest
            mean_cum_simple_regret_TT=myYbest_cum
        else:
            #mean_TT.append(temp_mean)
            #mean_TT=np.vstack((mean_TT,myYbest))
            mean_cum_simple_regret_TT=np.vstack((mean_cum_simple_regret_TT,myYbest_cum))
            
    
    mean_cum_simple_regret_TT=np.array(mean_cum_simple_regret_TT)   
    std_cum_TT=np.std(mean_cum_simple_regret_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_simple_regret_TT=np.mean(mean_cum_simple_regret_TT,axis=0)
   
    #return mean_TT[::step],std_TT[::step]#,mean_cum_TT[::5],std_cum_TT[::5]
    #return mean_TT,std_TT,np.mean(mean_cum_simple_regret_TT),np.mea(std_cum_TT)
    
    #half_list_index=np.int(len(mean_cum_simple_regret_TT)*0.5)
    #return np.mean(mean_cum_simple_regret_TT[half_list_index:]),np.mean(std_cum_TT[half_list_index:])
    return np.mean(mean_cum_simple_regret_TT),np.mean(std_cum_TT)