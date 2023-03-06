# -*- coding: utf-8 -*-


import numpy as np
from collections import OrderedDict
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def plot(self):
        bounds=self.bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
                arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds=np.asarray(bounds)
        X=np.array([np.arange(x[0], x[1], 0.01) for x in arr_bounds])
        X=X.reshape(-1,2)
        X1=np.array([X[:,0]])
        X2=np.array([X[:,1]])
        X1, X2 = np.meshgrid(X1, X2)
        y=np.zeros([X1.shape[1],X2.shape[1]])
        #print(y.shape)
        #print(X1.shape)
        #print(X2.shape)
        for ii in range(0,X1.shape[1]):
            for jj in range(0,X2.shape[1]):
                Xij=np.array([X1[ii,ii],X2[jj,jj]])
                #print(Xij)
                y[ii,jj]=self.func(Xij)
                #        f1=plt.figure(1)
                #        ax=plt.axes(projection='3d')
                #        ax.plot_surface(X1,X2,y)
                plt.contourf(X1,X2,y,levels=np.arange(0,35,1))
                plt.colorbar()
        
    
    def findSdev(self):
        num_points_per_dim=100
        bounds=self.bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            keys = bounds.keys()
            arr_bounds = []
            for key in keys:
                arr_bounds.append(bounds[key])
            arr_bounds = np.asarray(arr_bounds)
        else:
            arr_bounds=np.asarray(bounds)
        X=np.array([np.random.uniform(x[0], x[1], size=num_points_per_dim) for x in arr_bounds])
        X=X.reshape(num_points_per_dim,-1)
        y=self.func(X)
        sdv=np.std(y)
        #maxima=np.max(y)
        #minima=np.min(y)
        return sdv
    
class saddlepoint(functions):
    def __init__(self):
        self.input_dim=2
        self.bounds=OrderedDict({'x1':(-1,1),'x2':(-1,1)})
        self.fstar=0
        self.min=0
        self.ismax=1
        self.name='saddlepoint'
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        fval=X[:,0]*X[:,0]-X[:,1]*X[:,1]
        return fval*self.ismax
        
class sin(functions):
    def __init__(self,sd=None):
        self.input_dim=1
        self.bounds={'x':(-1,15)}
        #self.bounds={'x':(0,1)}

        self.fstar=11
        self.min=0
        self.ismax=1
        self.name='sin'
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
    def func(self,x):
        x=np.asarray(x)

        fval=np.sin(x)
        return fval*self.ismax
        
class sincos(functions):
    def __init__(self,sd=None):
        self.input_dim=1
        self.bounds={'x':(-1,2)}
        #self.bounds={'x':(0,1)}

        self.fstar=11
        self.min=0
        self.ismax=1
        self.name='sincos'
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
    def func(self,x):
        x=np.asarray(x)

        fval=x*np.sin(x)+x*np.cos(2*x)
        return fval*self.ismax

class fourier(functions):
    '''
    Forrester function. 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,sd=None):
        self.bounds = {'x':(0,10)}
        self.sd=0
        
        self.input_dim = 1
        self.ismax=-1
        

        self.min = 4.795         ## approx
        self.fstar = -9.5083483926941064*self.ismax             ## approx

        self.name='fourier'
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
        
    def func(self,X):
        X=np.asarray(X)
        X = X.reshape((len(X),1))
        n = X.shape[0]
        fval = X*np.sin(X)+X*np.cos(2*X)
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
        return self.ismax*fval.reshape(n,1) + noise
        
        
class branin(functions):
    def __init__(self,sd=None):
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
            
        self.input_dim=2
        #if sd==None: self.sd = 0
        #else: self.sd=sd
        self.bounds=OrderedDict([('x1',(-5,10)),('x2',(0,15))])
        #self.bounds=OrderedDict([('x1',(-20,70)),('x2',(-50,50))])
        self.ismax=-1
        self.fstar=0.397887*self.ismax
        self.min=[9.424,2.475]

        self.name='branin'
        
        
    def func(self,X):
        
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        n=X.shape[0]

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        a=1
        b=5.1/(4*np.pi**2)
        c=5/np.pi
        r=6
        s=10
        t=1/(8*np.pi)
        fx=a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s    
        
        
        if self.sd==0:
            return fx*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fx*self.ismax+np.ravel(noise)
        

        
class forrester(functions):
    '''
    Forrester function. 
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, sd=None):     
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
        self.ismax=-1
        
        self.input_dim = 1        
        self.min = 0.78         ## approx
        self.fstar = -6.03*self.ismax             ## approx
        self.bounds = {'x':(0,1)}
        self.name='forrester'
        #self.sd=0
      
    def func(self,X):
        X=np.asarray(X)
        X = X.reshape((len(X),1))
        n = X.shape[0]

        fval = ((6*X -2)**2)*np.sin(12*X-4)
        if self.sd!=0:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)
        else:
            return fval*self.ismax



class rosenbrock(functions):
    '''
    rosenbrock function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0):
        if sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-2.048,2.048)),('x2',(-2.048,2.048))])
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.ismax=-1

        self.fstar = 0
        
        self.name = 'Rosenbrock'
        #self.sd=self.findSdev()
        
    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        n=X.shape[0]
        n=1
        if len(X.shape)==1:# one observation
            x1=X[0]
            x2=X[1]
        else:# multiple observations
            x1=X[:,0]
            x2=X[:,1]
            n=X.shape[0]
        fx = 100*(x2-x1**2)**2 + (x1-1)**2
        if self.sd==0:
            return fx*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fx*self.ismax+np.ravel(noise)
   


class beale(functions):
    '''
    beale function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        if sd==None:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict({'x1':(-1,1),'x2':(-1,1)})
        else: self.bounds = bounds
        self.min = [(3, 0.5)]
        self.ismax=-1

        self.fstar = 0
        self.name = 'Beale'



    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]    
        fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise) 
            


class dropwave(functions):
    '''
    dropwave function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        if sd==None:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-5.12,5.12)),('x2',(-5.12,5.12))])
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.ismax=-1
        self.fstar = -1*self.ismax

        self.name = 'dropwave'

        
    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        n=1
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        
        fval = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2)+2) 
        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)


class cosines(functions):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(0,1)),('x2',(0,1))])
        else: self.bounds = bounds
        self.min = [(0.31426205,  0.30249864)]
        self.ismax=1
        self.fstar = -1.59622468*self.ismax

        self.name = 'Cosines'

    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        #X = reshape(X,self.input_dim)
        #n = X.shape[0]
        
        u = 1.6*x1-0.5
        v = 1.6*x2-0.5
        fval = 1-(u**2 + v**2 - 0.3*np.cos(3*np.pi*u) - 0.3*np.cos(3*np.pi*v) )

        return self.ismax*fval
            
            
            
class goldstein(functions):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-2,2),'x2':(-2,2)}
        else: self.bounds = bounds
        self.ismax=-1
        self.min = [(0,-1)]
        self.fstar = 3*self.ismax
        self.name = 'Goldstein'

    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
   
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a*fact1b
        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a*fact2b
        fval = fact1*fact2

        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)



class sixhumpcamel(functions):
    '''
    Six hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-3,3)),('x2',(-2,2))])
        else: self.bounds = bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.ismax=-1
        self.fstar = -1.0316*self.ismax
        self.name = 'Six-hump camel'
          
    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        n=1
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)


class mccormick(functions):
    '''
    Mccormick function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1.5,4),(-3,4)]
        else: self.bounds = bounds
        self.min = [(-0.54719,-1.54719)]
        self.ismax=-1
        self.fstar = -1.9133*self.ismax        
        self.name = 'Mccormick'

    def func(self,X):
        X = reshape(X,self.input_dim)

        x1=X[:,0]
        x2=X[:,1]
 
      
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2)**2
        term3 = -1.5*x1
        term4 = 2.5*x2
        fval = term1 + term2 + term3 + term4 + 1
        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)


class powers(functions):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fstar = 0
         #if sd==None: self.sd = 0
         #else: self.sd=sd
        self.name = 'Sum of Powers'

    def func(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            fval = abs(x1)**2 + abs(x2)**3
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class eggholder(functions):
    def __init__(self,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            #self.sd=self.findSdev()
            self.sd=sd
        self.input_dim = 2
        #self.bounds = {'x1':(-512,512),'x2':(-512,512)}
        self.bounds = [(-512,512),(-512,512)]      
        self.min = [(512,404.2319)]
        self.ismax=-1
        self.fstar = -959.6407*self.ismax       
        self.name = 'Egg-holder'

    def func(self,X):

        X=np.asarray(X)
        X = reshape(X,self.input_dim)

        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
            
        fval = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
        n=X.shape[0]
        
        if self.sd==0:
            return fval*self.ismax
        else:
            noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
            return fval*self.ismax+np.ravel(noise)

class alpine1(functions):
    '''
    Alpine1 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self,input_dim,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
            
        if bounds == None: 
            self.bounds = bounds  =[(-10,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.input_dim = input_dim
            
        self.ismax=-1
        #self.fstar = -46*self.ismax
        self.fstar = 0*self.ismax

        self.name='alpine1'

        
    def func(self,X):
        X = reshape(X,self.input_dim)
        #n = X.shape[0]
        temp=np.abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape)<=1:
            fval=np.sum(temp)
        else:
            fval = np.sum(temp,axis=1)
        n=X.shape[0]
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return self.ismax*fval.reshape(n,1) + noise


class alpine2(functions):
    '''
    Alpine2 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,input_dim,bounds=None,sd=0):
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=self.findSdev()
        if bounds == None: 
            self.bounds = bounds  =[(1,10)]*input_dim
        else: 
            self.bounds = bounds
        

        self.min = [(7.917)]*input_dim
        self.ismax=-1
        self.fstar = self.ismax*(-2.808**input_dim)
        self.input_dim = input_dim

        self.name='Alpine2'
    
        
    def internal_func(self,X):
        fval = np.cumprod(np.sqrt(X))[self.input_dim-1]*np.cumprod(np.sin(X))[self.input_dim-1]  
        #fval = np.cumprod(np.sqrt(X)*np.sin(X))
        return fval

    def func(self,X):
        X=np.asarray(X)
        X = reshape(X,self.input_dim)
        #n = X.shape[0]
        #fval = np.cumprod(np.sqrt(X),axis=1)[:,self.input_dim-1]*np.cumprod(np.sin(X),axis=1)[:,self.input_dim-1]  
        #fval = np.cumprod(np.sqrt(X))[:,self.input_dim-1]*np.cumprod(np.sin(X))[:,self.input_dim-1]
        fval=[self.ismax*self.internal_func(val) for idx, val in enumerate(X)]
        fval=np.asarray(fval)
        #noise = np.random.normal(0,0.1*self.sd,n).reshape(n,1)
        n=X.shape[0]
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return self.ismax*fval.reshape(n,1) + noise

class gSobol(functions):
    '''
    gSolbol function
   
    :param a: one-dimensional array containing the coefficients of the function.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,a,bounds=None,sd=None):
        self.a = a
        self.input_dim = len(self.a)

        if bounds == None: 
            self.bounds =[(-4,6)]*self.input_dim
        else: 
            self.bounds = bounds

        if not (self.a>0).all(): return 'Wrong vector of coefficients, they all should be positive'
        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd==None: self.sd = 0
        else: self.sd=sd

        self.ismax=-1
        self.fstar=0# in correct
        #self.fstar=20*self.ismax# in correct
        self.name='gSobol'

    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n,1)*self.a)/(1+np.ones(n).reshape(n,1)*self.a)
        fval =  np.cumprod(aux,axis=1)[:,self.input_dim-1]

        n=X.shape[0]
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return self.ismax*fval.reshape(n,1) + noise

#####
class ackley(functions):
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=sd
            #self.sd=self.findSdev()
            
        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fstar = 0
        self.ismax=-1
        self.name='ackley'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        #print X
        #n = X.shape[0]
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))
      
        n=X.shape[0]
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return self.ismax*fval.reshape(n,1) + noise


#####
class hartman_6d:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=sd
            #self.sd=self.findSdev()
            
        self.input_dim = 6

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.ismax=-1

        #self.fstar = -3.32237*self.ismax
        self.fstar = -3.05*self.ismax

        self.name='hartman_6d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]]

        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(6):
                    xj = X[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -(2.58 + outer) / 1.94;
      
        noise = np.random.normal(0,self.sd,n).reshape(n,1)

        if n==1:
            return self.ismax*(fval[0][0])+noise
        else:
            return self.ismax*(fval)+noise
        
        
"""
class hartman_4d:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fstar = -3.32237
        self.ismax=-1
        self.name='hartman_4d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;
        

        fval  =np.zeros((n,1))        
        for idx in range(n):
            X_idx=X[idx,:]
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(4):
                    xj = X_idx[jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = (1.1 - outer) / 0.839;
        if n==1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
"""            
            
class hartman_3d(functions):
    '''
    hartman_3d: function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        
        if sd==None or sd==0:
            self.sd=0
        else:
            self.sd=sd
            #self.sd=self.findSdev()
            
        self.input_dim = 3
        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [0.114614,0.555649,0.852547]
        self.ismax=-1
        
        self.fstar = -3.86278*self.ismax
        #self.fstar = -3.7078*self.ismax

        self.name='hartman_3d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[3.0, 10, 30],
             [0.1, 10, 35],
             [3.0, 10, 30],
             [0.1, 10, 35]]
        A=np.asarray(A)
        P = [[3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]]

        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(3):
                    xj = X[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -outer;
        
        noise = np.random.normal(0,self.sd,n).reshape(n,1)
        #noise=0
        if n==1:
            return self.ismax*(fval[0][0])+noise
        else:
            return self.ismax*(fval)+noise
        
class mixture(functions):
    '''
    a scalable gaussian mixture function
    
    :param sd: standard deviation to generate noisy exaluations of the functions
    :param peaks: number of gaussian peaks used
    '''
    def __init__(self,bounds=None, peaks=3):
        self.input_dim=2
        self.peaks=peaks
        self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fstar=-1
        self.ismax=-1
        self.name="mixture"
        self.sd=self.findSdev()
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        y=2*multivariate_normal.pdf(X,mean=[0.5,0.5],cov=0.07*np.eye(2))
        if self.peaks>=2:
            y+=1.8*multivariate_normal.pdf(X,mean=[0.2,0.2],cov=0.03*np.eye(2))
        if self.peaks>=3:
            y+=1.7*multivariate_normal.pdf(X,mean=[0.7,0.7],cov=0.07*np.eye(2))
        if self.peaks>=4:
            y+=1*multivariate_normal.pdf(X,mean=[0.8,0.5],cov=0.02*np.eye(2))
        if self.peaks>=5:
            y+=1.7*multivariate_normal.pdf(X,mean=[0.4,0.6],cov=0.005*np.eye(2))
        if self.peaks>=6:
            y+=1.75*multivariate_normal.pdf(X,mean=[0.3,0.4],cov=0.0012*np.eye(2))
        if self.peaks>=7:
            y+=1.75*multivariate_normal.pdf(X,mean=[0.9,0.8],cov=0.01*np.eye(2))
        if self.peaks>=8:
            y+=1.75*multivariate_normal.pdf(X,mean=[0.2,0.6],cov=0.01*np.eye(2))
        if self.peaks>=9:
            y+=1.75*multivariate_normal.pdf(X,mean=[0.9,0.3],cov=0.01*np.eye(2))
        return y
    
class gaussian(functions):
    '''
    a scalable gaussian mixture function
    
    :param sd: standard deviation to generate noisy exaluations of the functions
    :param peaks: number of gaussian peaks used
    '''
    def __init__(self,bounds=None, dim=3):
        self.input_dim=dim
        self.sd=0
        if bounds == None:
            self.bounds =[(0,1)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.)*self.input_dim]
        self.fstar=-1
        self.ismax=-1
        self.name="gaussian"
        self.sd=self.findSdev()
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        noise = np.random.normal(0,self.sd,n).reshape(n,1)
        y=multivariate_normal.pdf(X,mean=0.5*np.ones(self.input_dim),cov=np.eye(self.input_dim))
        return y