

#from sklearn.gaussian_process import GaussianProcess
#from scipy.stats import norm
#import matplotlib as plt

#from mpl_toolkits.mplot3d import Axes3D


import numpy as np
#import random
import time
#import os

from tqdm import tqdm

                                    

def run_experiment(bo,gp_params,yoptimal=0,n_init=3,NN=10,runid=1):
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init,seed=runid)
    
    # number of recommended parameters
    for index in range(0,NN):
        bo.maximize()
        

    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return fxoptimal, elapsed_time

    
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0,step=3):
    
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
            
#def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0,step=3):
#    
#    nRepeat=len(YY)
#    YY=np.asarray(YY)
#    print(YY.shape)
#    mean_TT=[]
#    
#    mean_cum_TT=[]
#    
#    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
#    
#        if IsPradaBO==1:
#            temp_mean=YY[idxtt,0:BatchSzArray[0]].max()
#        else:
#            temp_mean=YY[idxtt,0:BatchSzArray[0]].min()
#        
#        temp_mean_cum=YY[idxtt,0:BatchSzArray[0]].mean()
#
#        start_point=0
#        for idx,bz in enumerate(BatchSzArray): # batch
#            if idx==len(BatchSzArray)-1:
#                break
#            bz=np.int(bz)
#
#            #    get the average in this batch
#            temp_mean_cum=np.vstack((temp_mean_cum,YY[idxtt,start_point:start_point+bz].mean()))
#            
#            # find maximum in each batch            
#            if IsPradaBO==1:
#                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].max()))
#            else:
#                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].min()))
#
#            start_point=start_point+bz
#
#        if IsPradaBO==1:
#            myYbest=[temp_mean[:idx+1].max()*-1 for idx,val in enumerate(temp_mean)]
#            temp_mean_cum=temp_mean_cum*-1
#            temp_mean=temp_mean*-1
#        else:
#            myYbest=[temp_mean[:idx+1].min() for idx,val in enumerate(temp_mean)]
#
#        
#        temp_regret=np.abs(temp_mean-Y_optimal)
#        myYbest_cum=[np.mean(temp_regret[:idx+1]) for idx,val in enumerate(temp_regret)]
#
#
#        if len(mean_TT)==0:
#            mean_TT=myYbest
#            mean_cum_TT=myYbest_cum
#        else:
#            #mean_TT.append(temp_mean)
#            mean_TT=np.vstack((mean_TT,myYbest))
#            mean_cum_TT=np.vstack((mean_cum_TT,myYbest_cum))
#            
#    mean_TT    =np.array(mean_TT)
#    std_TT=np.std(mean_TT,axis=0)
#    std_TT=np.array(std_TT).ravel()
#    mean_TT=np.mean(mean_TT,axis=0)
#
#    
#    mean_cum_TT=np.array(mean_cum_TT)   
#    std_cum_TT=np.std(mean_cum_TT,axis=0)
#    std_cum_TT=np.array(std_cum_TT).ravel()
#    mean_cum_TT=np.mean(mean_cum_TT,axis=0)
#   
#    
#    print(step)
#    return mean_TT[::step],std_TT[::step],mean_cum_TT[::step],std_cum_TT[::step]
    
