import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from bayes_opt import BayesOpt,BayesOpt_KnownOptimumValue


import numpy as np
#from bayes_opt import auxiliary_functions
from bayes_opt import functions
from bayes_opt import utilities
import warnings
#from bayes_opt import acquisition_maximization

import sys

import itertools


import matplotlib.pyplot as plt


np.random.seed(6789)

warnings.filterwarnings("ignore")


counter = 0


myfunction_list=[]

#myfunction_list.append(functions.sincos())
#myfunction_list.append(functions.branin())
#myfunction_list.append(functions.hartman_3d())
#myfunction_list.append(functions.ackley(input_dim=5))
myfunction_list.append(functions.alpine1(input_dim=5))
#myfunction_list.append(functions.hartman_6d())
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1])))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1])))


acq_type_list=[]


temp={}
temp['name']='erm' # expected regret minimization
temp['IsTGP']=0 # recommended to use tgp for ERM
acq_type_list.append(temp)

temp={}
temp['name']='cbm' # confidence bound minimization
temp['IsTGP']=1 # recommended to use tgp for CBM
#acq_type_list.append(temp)


#temp={}
#temp['name']='kov_mes' # MES+f*
#temp['IsTGP']=0 # we can try 'tgp'
#acq_type_list.append(temp)



temp={}
temp['name']='kov_ei' # this is EI + f*
temp['IsTGP']=0 # we can try 'tgp' by setting it =1
#acq_type_list.append(temp)



temp={}
temp['name']='ucb' # vanilla UCB
temp['IsTGP']=0 # we can try 'tgp' by setting it =1
#acq_type_list.append(temp)



temp={}
temp['name']='ei' # vanilla EI
temp['IsTGP']=0 # we can try 'tgp' by setting it =1
#acq_type_list.append(temp)



temp={}
temp['name']='random' # vanilla EI
temp['IsTGP']=0 # we can try 'tgp' by setting it =1
#acq_type_list.append(temp)

fig=plt.figure()

color_list=['r','b','k','m','c','g','o']
marker_list=['s','x','o','v','^','>','<']

for idx, (myfunction,acq_type,) in enumerate(itertools.product(myfunction_list,acq_type_list)):

    print("=====================func:",myfunction.name)
    print("==================acquisition type",acq_type)
    
    IsTGP=acq_type['IsTGP']
    acq_name=acq_type['name']
    
    nRepeat=10
    
    ybest=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat
    marker=[0]*nRepeat

    bo=[0]*nRepeat
   
    [0]*nRepeat
    
    for ii in range(nRepeat):
        
        if 'kov' in acq_name or acq_name == 'erm' or acq_name == 'cbm':
            bo[ii]=BayesOpt_KnownOptimumValue(myfunction.func,myfunction.bounds,myfunction.fstar, \
                                  acq_name,IsTGP,verbose=1)
        else:
            bo[ii]=BayesOpt(myfunction.func,myfunction.bounds,acq_name,verbose=1)
  
        ybest[ii],MyTime[ii]=utilities.run_experiment(bo[ii],n_init=3*myfunction.input_dim,\
             NN=10*myfunction.input_dim,runid=ii)        
                                       
        MyOptTime[ii]=bo[ii].time_opt
        print("ii={} BFV={:.3f}".format(ii,myfunction.ismax*np.max(ybest[ii])))                                              
        

    Score={}
    Score["ybest"]=ybest
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    
    utilities.print_result_sequential(bo,myfunction,Score,acq_type) 
    
    
    ## plot the result
        
    # process the result
    
    y_best_sofar=[0]*len(bo)
    for uu,mybo in enumerate(bo):
        y_best_sofar[uu]=[ (myfunction.fstar - np.max(mybo.Y_ori[:ii+1]) ) for ii in range(len(mybo.Y_ori))]
        y_best_sofar[uu]=y_best_sofar[uu][3*myfunction.input_dim:] # remove the random phase for plotting purpose
        
    y_best_sofar=np.asarray(y_best_sofar)
    
    myxaxis=range(y_best_sofar.shape[1])
    
    plt.errorbar(myxaxis,np.mean(y_best_sofar,axis=0), np.std(y_best_sofar,axis=0)/np.sqrt(nRepeat),
                 label=acq_type['name'],color=color_list[idx],marker=marker_list[idx])
    
    
    
plt.ylabel("Simple Regret",fontsize=14)
plt.xlabel("Iterations",fontsize=14)
plt.legend(prop={'size': 14})
strTitle="{:s} D={:d}".format(myfunction.name,myfunction.input_dim)
plt.title(strTitle,fontsize=18)
