

import itertools
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
#from scipy.optimize import fmin_bfgs
#from scipy.optimize import fmin_l_bfgs_b
#from sklearn.metrics.pairwise import euclidean_distances
from bayes_opt.acquisition_functions import AcquisitionFunction

#from scipy.optimize import fmin_cobyla
#from helpers import UtilityFunction, unique_rows
#from visualization import Visualization
#from prada_gaussian_process import PradaGaussianProcess
#from prada_gaussian_process import PradaMultipleGaussianProcess
import random
import time
#from sortedcontainers import SortedList
#from ..util.general import multigrid, samples_multidimensional_uniform, reshape
import sobol_seq



def acq_max_with_name(gp,scalebounds,acq_name="ei",IsReturnY=False,IsMax=True,fstar_scaled=None):
    acq={}
    acq['name']=acq_name
    acq['dim']=scalebounds.shape[0]
    acq['scalebounds']=scalebounds   
    if fstar_scaled:
        acq['fstar_scaled']=fstar_scaled   

    myacq=AcquisitionFunction(acq)
    if IsMax:
        x_max = acq_max(ac=myacq.acq_kind,gp=gp,bounds=scalebounds,opt_toolbox='scipy')
    else:
        x_max = acq_min_scipy(ac=myacq.acq_kind,gp=gp,bounds=scalebounds)
    if IsReturnY==True:
        y_max=myacq.acq_kind(x_max,gp=gp)
        return x_max,y_max
    return x_max


def acq_max_nlopt(ac,gp,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'NLOPT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    
    y_max=np.max(gp.Y)
    try:
        import nlopt
    except:
        print("Cannot find nlopt library")
    
    
    def objective(x, grad):
        if grad.size > 0:
            print("here grad")
            fx, gx = ac(x[None], grad=True)
            grad[:] = gx[0][:]

        else:

            fx = ac(x,gp)
            fx=np.ravel(fx)
            #print fx
            if isinstance(fx,list):
                fx=fx[0]
        #return np.float64(fx[0])
        return fx[0]

    tol=1e-7
    bounds = np.array(bounds, ndmin=2)

    dim=bounds.shape[0]
    #opt = nlopt.opt(nlopt.GN_DIRECT, dim)
    opt = nlopt.opt(nlopt.GN_DIRECT  , dim)
    #opt = nlopt.opt(nlopt.LN_BOBYQA , bounds.shape[0])

    opt.set_lower_bounds(bounds[:, 0])
    opt.set_upper_bounds(bounds[:, 1])
    #opt.set_ftol_rel(tol)
    opt.set_maxeval(1000*dim)
    #opt.set_xtol_abs(tol)

    #opt.set_ftol_abs(tol)#Set relative tolerance on function value.
    #opt.set_xtol_rel(tol)#Set absolute tolerance on function value.
    #opt.set_xtol_abs(tol) #Set relative tolerance on optimization parameters.

    opt.set_maxtime=1000*dim
    
    opt.set_max_objective(objective)    

    xinit=random.uniform(bounds[:,0],bounds[:,1])
    #xinit=np.asarray(0.2)
    #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
    #print xoptimal
    
    #try:
    xoptimal = opt.optimize(xinit.copy())

    #except:
        #xoptimal=xinit
        #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
     
    fmax= opt.last_optimum_value()
    
    #print "nlopt force stop ={:s}".format(nlopt_result)
    #fmax=opt.last_optimize_result()
    
    code=opt.last_optimize_result()
    status=1

    """
    if code==-1:
        print 'NLOPT fatal error -1'
        status=0
        """    

    if code<0:
        print("nlopt code = {:d}".format(code))
        status=0


    return xoptimal, fmax, status

def acq_max_scipydirect(ac,gp,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'DIRECT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    try:
        from scipydirect import minimize
    except:
        print("Cannot find scipydirect library")
    
    myfunc=lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=np.max(gp.Y))
    res = minimize(func=myfunc, bounds=bounds)
    return np.reshape(res,len(bounds))


def acq_max_direct(ac,gp,y_max,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'DIRECT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    
    try:
        from DIRECT import solve
    except:
        print("Cannot find DIRECT library")
        
    def DIRECT_f_wrapper(ac):
        def g(x, user_data):
            fx=ac(np.array([x]),gp,y_max)
            #print fx[0]
            return fx[0], 0
        return g
            
    lB = np.asarray(bounds)[:,0]
    uB = np.asarray(bounds)[:,1]
    
    #x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=750, maxf=2000,volper=0.005) # this can be used to speed up DIRECT (losses precission)
    x,_,_ = solve(DIRECT_f_wrapper(ac),lB,uB,maxT=2000,maxf=2000,volper=0.0005)
    return np.reshape(x,len(bounds))


idx_tracing=0
smallest_y=0
smallest_y_index=0
flagReset=0

def acq_max_with_tracing(ac,gp,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'DIRECT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    # number of candidates

    global idx_tracing
    global smallest_y
    global smallest_y_index
    idx_tracing=0
    smallest_y=0
    smallest_y_index=0
    
    nCandidates=50*gp.X.shape[1]
    #nCandidates=5
    
    myXList=[0]*nCandidates
    myYList=[0]*nCandidates
        
    try:
        import nlopt
    except:
        print("Cannot find nlopt library")
    
    
    def objective(x, grad):
        if grad.size > 0:
            print("here grad")
            fx, gx = ac(x[None], grad=True)
            grad[:] = gx[0][:]

        else:

            fx = ac(x,gp)
            fx=np.ravel(fx)
            #print fx
            if isinstance(fx,list):
                fx=fx[0]

            global idx_tracing
            global smallest_y
            global smallest_y_index
            if idx_tracing<nCandidates-1: # if the list is still empty
                myXList[idx_tracing]=np.copy(x)
                myYList[idx_tracing]=np.copy(fx[0])
                idx_tracing=idx_tracing+1
            #elif idx_tracing==nCandidates-1:
                #myXList[idx_tracing]=np.copy(x)
                #myYList[idx_tracing]=np.copy(fx[0])
                #idx_tracing=idx_tracing+1
                smallest_y_index=np.argmin(myYList)
                smallest_y=myYList[smallest_y_index]
            elif fx > smallest_y: # find better point
                #if fx > smallest_y: # find better point
                    myXList[smallest_y_index]=np.copy(x)
                    myYList[smallest_y_index]=np.copy(fx[0])
                    # update max_y
                    smallest_y_index=np.argmin(myYList)
                    smallest_y=myYList[smallest_y_index]
            #print myYList
            #print myXList
        return fx[0]

            
    tol=1e-7
    bounds = np.array(bounds, ndmin=2)

    dim=bounds.shape[0]
    #opt = nlopt.opt(nlopt.GN_DIRECT, dim)
    opt = nlopt.opt(nlopt.GN_DIRECT  , dim)
    #opt = nlopt.opt(nlopt.LN_BOBYQA , bounds.shape[0])

    opt.set_lower_bounds(bounds[:, 0])
    opt.set_upper_bounds(bounds[:, 1])
    #opt.set_ftol_rel(tol)
    opt.set_maxeval(500*dim)
    opt.set_xtol_abs(tol)

    opt.set_ftol_abs(tol)#Set relative tolerance on function value.
    #opt.set_xtol_rel(tol)#Set absolute tolerance on function value.
    #opt.set_xtol_abs(tol) #Set relative tolerance on optimization parameters.

    opt.set_maxtime=500*dim
    
    opt.set_max_objective(objective)    

    xinit=random.uniform(bounds[:,0],bounds[:,1])
    #xinit=np.asarray(0.2)
    #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
    #print xoptimal
    
    xoptimal = opt.optimize(xinit.copy())

    
    code=opt.last_optimize_result()
    status=1

    """
    if code==-1:
        print 'NLOPT fatal error -1'
        status=0
        """    

    if code<0:
        print("nlopt code = {:d}".format(code))
        status=0


    #reset the global variable

    return xoptimal, myXList, myYList
    #return np.reshape(x,len(bounds)), myXList, myYList

    
def acq_max(ac, gp, bounds, opt_toolbox='scipy',seeds=[],IsMax=True):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    y_max=np.max(gp.Y)
    if opt_toolbox=='nlopt':
        x_max,f_max,status = acq_max_nlopt(ac=ac,gp=gp,bounds=bounds)
        
        if status==0:# if nlopt fails, let try scipy
            opt_toolbox='scipy'
            
    if opt_toolbox=='direct':
        x_max = acq_max_direct(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    elif opt_toolbox=='scipydirect':
        x_max = acq_max_scipydirect(ac=ac,gp=gp,bounds=bounds)
    elif opt_toolbox=='scipy':
        x_max = acq_max_scipy(ac=ac,gp=gp,bounds=bounds)
    elif opt_toolbox=='thompson': # thompson sampling
        x_max = acq_max_thompson(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    elif opt_toolbox=='cobyla':
        x_max = acq_max_cobyla(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    elif opt_toolbox=='local_search':
        x_max = acq_max_local_search(ac=ac,gp=gp,y_max=y_max,bounds=bounds,seeds=seeds)
    return x_max

def generate_sobol_seq(dim,nSobol):
    mysobol_seq = sobol_seq.i4_sobol_generate(dim, nSobol)
    return mysobol_seq
    
    
    
def acq_max_geometric(ac, gp, bounds,cache_sobol):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}
   
    # create a grid
    #XXX = np.meshgrid(*[np.linspace(i,j,10)[:-1] for i,j in zip(bounds[:,0],bounds[:,1])])
    
    ninitpoint=200*dim
    #ninitpoint=5*dim
    
    if cache_sobol is not None:
        x_tries=cache_sobol

    else:
        print('sobol sequence is not cached')
        x_tries = sobol_seq.i4_sobol_generate(dim, ninitpoint)
        

    # randomly select points and evaluate points from a grid
    
    
    # Find the minimum of minus the acquisition function        
    #x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))

    # evaluate
    #start_eval=time.time()
    y_tries=ac(x_tries,gp=gp)
    #end_eval=time.time()
    #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
    
    #find x optimal for init
    idx_max=np.argmax(y_tries)
    x_max=x_tries[idx_max]
    
    #start_opt=time.time()

    #res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
    
    
    #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
    # value at the estimated point
    #val=ac(res.x,gp,y_max)        
   

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

def acq_min_scipy(ac, gp, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    min_acq = None

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':10*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}

    # multi start
    for i in range(5*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp)
        
        #find x optimal for init
        idx_max=np.argmin(y_tries)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 
        
        # Store it if better than previous minimum(maximum).
        if min_acq is None or val <= min_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            min_acq = val
            #print max_acq

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    
def acq_max_scipy(ac, gp, bounds):
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

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':10*dim,'maxfun':20*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in range(10*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(10*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp)
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max

def acq_max_thompson(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in range(5*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp)
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    
def acq_max_with_init(ac, gp, y_max, bounds, init_location=[]):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in range(2*dim):
        # Find the minimum of minus the acquisition function 
        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
        
        if init_location!=[]:
            x_tries=np.vstack((x_tries,init_location))
        
            
        y_tries=ac(x_tries,gp=gp)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp)        
        else:
            val=ac(res.x,gp) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def acq_max_local_search(ac, gp, y_max, bounds,seeds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    dim=bounds.shape[0]

    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    #myopts ={'maxiter':100,'maxfun':10*dim}
    myopts ={'maxiter':5*dim}

    myidx=np.random.permutation(len(seeds))
    # multi start
    for idx in range(5*dim):
    #for i in xrange(1*dim):
    #for idx,xt in enumerate(seeds): 
        xt=seeds[myidx[idx]]
        val=ac(xt,gp,y_max) 
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val > max_acq:
            x_max=xt
            max_acq = val
        #for i in xrange(1*dim):
        for i in range(1):
            res = minimize(lambda x: -ac(x, gp=gp, y_max=y_max),xt,bounds=bounds,
                           method="L-BFGS-B",options=myopts)#L-BFGS-B
    
            xmax_temp=np.clip(res.x, bounds[:, 0], bounds[:, 1])
            val=ac(xmax_temp,gp,y_max) 

            # Store it if better than previous minimum(maximum).
            if max_acq is None or val > max_acq:
                x_max = xmax_temp
                max_acq = val
                #print max_acq

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    

def acq_max_single_seed(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    #max_acq = None
    dim=bounds.shape[0]

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50*dim, dim))
    
    # evaluate
    y_tries=ac(x_tries,gp=gp, y_max=y_max)
        
    #find x optimal for init
    idx_max=np.argmax(y_tries)
    x_init_max=x_tries[idx_max]
    #x_try=np.array(bounds[:, 0])

    # Find the minimum of minus the acquisition function
    res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                   x_init_max.reshape(1, -1),
                   bounds=bounds,
                   method="L-BFGS-B")

    x_max = res.x
    #max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
