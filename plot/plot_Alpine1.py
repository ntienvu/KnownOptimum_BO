# -*- coding: utf-8 -*-


import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../../')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bayes_opt import auxiliary_functions



sns.set(style="ticks")

fig=plt.figure(figsize=(10, 6))


##############
function_name='Alpine1'
D=5
optimal_value=-29

BatchSz0=D*3

start_point=BatchSz0
step=5
mylinewidth=2
alpha_level=0.3
std_scale=0.2

T=10
#BatchSz_GPyOpt=[D]*(1+D*10)
BatchSz=[1]*(D*T+1)
BatchSz[0]=3*D


x_axis=np.array(range(0,D*T+1))
x_axis=x_axis[::step]

# is minimization problem
IsMin=1
#IsMin=-1
IsLog=1


# UCB  
strFile="../pickle_storage/{:s}_{:d}_ucb_gp.pickle".format(function_name,D)
with open(strFile, 'rb') as f:
    UCB = pickle.load(f,encoding='bytes')
    
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(UCB[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)
myYbest=myYbest-optimal_value


if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale
plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='m',linestyle=':',marker='v', label='GP-UCB')




#EI
strFile="../pickle_storage/{:s}_{:d}_ei_gp.pickle".format(function_name,D)
with open(strFile,'rb') as f:
    EI = pickle.load(f,encoding='bytes')
    

myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(EI[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)
myYbest=IsMin*np.asarray(myYbest)


myYbest=myYbest-optimal_value

if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale

plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='r',linestyle='-.',marker='h', label='EI')






# EI + f*
strFile="../pickle_storage/{:s}_{:d}_kov_ei_gp.pickle".format(function_name,D)
print(strFile)
with open(strFile,'rb') as f:
    #[ybest, Regret, MyTime]
    KOV_EI = pickle.load(f,encoding='bytes')    
    
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(KOV_EI[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)
myYbest=myYbest-optimal_value

if IsLog==1:
    if any(x<0 for x in myYbest.tolist()):
        myYbest=-1*np.log(abs(myYbest))
    else:
        myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale
plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='g',linestyle='-',marker='>',label='EI+$f^*$')

    



# KOV MES
strFile="../pickle_storage/{:s}_{:d}_kov_mes_gp.pickle".format(function_name,D)
with open(strFile,'rb') as f:
    MES = pickle.load(f,encoding='bytes')
    
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(MES[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)


myYbest=IsMin*np.asarray(myYbest)
myYbest=myYbest-optimal_value

if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale
plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='y',linestyle='-',marker='s', label='MES+$f^*$')




# KOV CBM
strFile="../pickle_storage/{:s}_{:d}_kov_cbm_tgp.pickle".format(function_name,D)
with open(strFile, 'rb') as f:
    UCB = pickle.load(f,encoding='bytes')
    
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(UCB[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)
myYbest=myYbest-optimal_value


if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale
plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='k',linestyle='-.',marker='s', label='CBM+$f^*$')

    



#KOV ERM
strFile="../pickle_storage/{:s}_{:d}_kov_erm_tgp.pickle".format(function_name,D)

with open(strFile,"rb") as f:
    KOV = pickle.load(f,encoding='bytes')

# VRS Of Thompson Sampling
myYbest,myStd,myYbestCum,myStdCum=auxiliary_functions.yBest_Iteration(KOV[0],BatchSz,IsPradaBO=1,Y_optimal=optimal_value,step=step)

myYbest=IsMin*np.asarray(myYbest)
myYbest=myYbest-optimal_value

if IsLog==1:
    myYbest=np.log(myYbest)
    myStd=np.log(myStd)
myStd=myStd*std_scale
plt.errorbar(x_axis,myYbest,yerr=myStd,linewidth=mylinewidth,color='k',linestyle='-',marker='o',label='ERM+$f^*$')



plt.ylabel('Log of Simple Regret',fontdict={'size':20})

plt.xlabel('Iteration',fontdict={'size':20})

plt.legend(loc='middle right',prop={'size':22},ncol=2)


plt.xlim([-1,T*D+1])


strTitle="{:s} D={:d}".format(function_name,D)


plt.title(strTitle,fontdict={'size':24})

plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)

strFile="fig/{:s}_{:d}_kov.pdf".format(function_name,D)
plt.savefig(strFile, bbox_inches='tight')


