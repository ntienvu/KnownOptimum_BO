import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')


from bayes_opt.sequentialBO.bo_known_optimum_value import BayesOpt_KnownOptimumValue
from bayes_opt.sequentialBO.bayesian_optimization import BayesOpt

import numpy as np
from bayes_opt import auxiliary_functions

from bayes_opt.test_functions import functions,real_experiment_function
import warnings
#from bayes_opt import acquisition_maximization

import sys

from bayes_opt.utility import export_results
import itertools


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
temp['name']='kov_erm' # expected regret minimization
temp['surrogate']='tgp' # recommended to use tgp for ERM
acq_type_list.append(temp)

temp={}
temp['name']='kov_cbm' # confidence bound minimization
temp['surrogate']='tgp' # recommended to use tgp for CBM
acq_type_list.append(temp)


temp={}
temp['name']='kov_mes' # MES+f*
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)



temp={}
temp['name']='kov_ei' # this is EI + f*
temp['surrogate']='gp' # we can try 'tgp'
#acq_type_list.append(temp)



temp={}
temp['name']='ucb' # vanilla UCB
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)


temp={}
temp['name']='ei' # vanilla EI
temp['surrogate']='gp' # we can try 'tgp'
acq_type_list.append(temp)


temp={}
temp['name']='random' # vanilla EI
temp['surrogate']='gp' # we can try 'tgp'
#acq_type_list.append(temp)

fig=plt.figure()

color_list=['r','b','k','m','c','g','o']
marker_list=['s','x','o','v','^','>','<']

for idx, (myfunction,acq_type,) in enumerate(itertools.product(myfunction_list,acq_type_list)):
    func=myfunction.func
    
    func_params={}
    func_params['function']=myfunction

    gp_params = {'lengthscale':0.04*myfunction.input_dim,'noise_delta':1e-8,
                 'isZeroMean':True} # the lengthscaled parameter will be optimized

    yoptimal=myfunction.fstar*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim
    acq_type['debug']=0
    acq_type['fstar']=myfunction.fstar

    acq_params={}
    acq_params['optimize_gp']='maximize'#maximize
    acq_params['acq_func']=acq_type
    
    nRepeat=15
    
    ybest=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat
    marker=[0]*nRepeat

    bo=[0]*nRepeat
   
    [0]*nRepeat
    
    for ii in range(nRepeat):
        
        if 'kov' in acq_type['name']:
            bo[ii]=BayesOpt_KnownOptimumValue(gp_params,func_params,acq_params,verbose=0)
        else:
            bo[ii]=BayesOpt(gp_params,func_params,acq_params,verbose=0)
  
        ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(bo[ii],gp_params,
             n_init=3*myfunction.input_dim,NN=10*myfunction.input_dim,runid=ii)                                               
        MyOptTime[ii]=bo[ii].time_opt
        print("ii={} BFV={}".format(ii,myfunction.ismax*np.max(ybest[ii])))                                              
        

    Score={}
    Score["ybest"]=ybest
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    
    export_results.print_result_sequential(bo,myfunction,Score,acq_type) 
    
    
    ## plot the result
        
    # process the result
    
    y_best_sofar=[0]*len(bo)
    for uu,mybo in enumerate(bo):
        y_best_sofar[uu]=[np.max(mybo.Y_original[:ii+1]) for ii in range(len(mybo.Y_original))]
        y_best_sofar[uu]=y_best_sofar[uu][3*myfunction.input_dim:] # remove the random phase for plotting purpose
        
    y_best_sofar=np.asarray(y_best_sofar)
    
    myxaxis=range(y_best_sofar.shape[1])
    
    plt.errorbar(myxaxis,np.mean(y_best_sofar,axis=0), np.std(y_best_sofar,axis=0),
                 label=acq_type['name'],color=color_list[idx],marker=marker_list[idx])
    
    
    
plt.ylabel("Simple Regret",fontsize=14)
plt.xlabel("Iterations",fontsize=14)
plt.legend(prop={'size': 14})
strTitle="{:s} D={:d}".format(myfunction.name,myfunction.input_dim)
plt.title(strTitle,fontsize=18)
