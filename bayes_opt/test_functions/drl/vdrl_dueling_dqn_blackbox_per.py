# -*- coding: utf-8 -*-


#from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning

import sys

sys.path.append('../')


import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from collections import Counter
#from vdrl_utilities import HiddenLayer
from sklearn.metrics.pairwise import euclidean_distances
import scipy.linalg as spla

#sys.path.append('drl')
from bayes_opt.test_functions.drl.prioritized_experience_replay import Memory

# so you can test different architectures
class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True,zeros=False):
        
        initializer = tf.contrib.layers.xavier_initializer()

        if zeros:
            self.W = np.zeros((M1, M2), dtype=np.float32)
        else:
            #self.W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
            self.W = tf.Variable(initializer(shape=[M1,M2]))
      
        #self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        
        self.params = [self.W]

        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(initializer(shape=[M2]))
            self.params.append(self.b)

            #self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)
    
class DQNModel:
    def __init__(self, D,K, hidden_layer_sizes, gamma,
                 max_experiences=50000, min_experiences=2000, batch_sz=32,memory_size=2000,
                 lr=1e-3):
        #K : action size
        #D: state_size
        #batch_size: draw from experience replay
        
        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.ISWeights_=tf.placeholder(tf.float32, shape=(None,1), name='ISW')
        
        self.memory= Memory(memory_size)

        self.K=K
        self.layers=[]
        M1=D
        for M2 in hidden_layer_sizes:
            layer=HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1=M2
            
        # calculate output and cost
        Z=self.X
        for layer in self.layers:
            Z=layer.forward(Z)
            
            
            
        ## Here we separate into two streams
        # The one that calculate V(s)
        self.value_fc = tf.layers.dense(inputs = Z,
                              units = 512,  activation = tf.nn.elu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="value_fc")
        
        self.value = tf.layers.dense(inputs = self.value_fc,
                                    units = 1, activation = None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="value")
        
        # The one that calculate A(s,a)
        self.advantage_fc = tf.layers.dense(inputs = Z,
                              units = 512,  activation = tf.nn.elu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="advantage_fc")
        
        self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                    units = K, activation = None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="advantages")
            
            
        # Agregating layer
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
          
        # Q is our predicted Q value.
        #self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
    
        self.predict_op = self.output
        
        selected_action_values = tf.reduce_sum(
                self.output * tf.one_hot(self.actions, K),
                reduction_indices=[1])
        
        
        #self.loss = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.loss = tf.reduce_sum(self.ISWeights_*tf.square(self.G - selected_action_values))

        
        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        #self.loss = tf.reduce_mean(tf.square(self.G - self.Q))
            
        # The loss is modified because of PER 
        self.absolute_errors = tf.abs(selected_action_values - self.G)# for updating Sumtree
        
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

        # create replay memory
        self.experience={'s':[],'a':[],'r':[],'s2':[],'done':[]}
        self.max_experiences=max_experiences
        self.min_experiences=min_experiences
        self.batch_sz=batch_sz
        self.gamma=gamma
        self.count_exp=0
        
        self.selected_exp=np.empty((0,D))
        self.max_selected_exp=self.batch_sz*20
        
        self.count_er_index = Counter()
        
        #offset will record the number of  deleting elements
        # in experience buffer
        self.offset=0
        
        
        self.logdet_selected_exp=[]
                
        
    def set_session(self, session):
        self.session = session
    
    def copy_from(self, other):
        # collect all the ops
        ops=[]
        for p,q in zip(self.params, other.params):
            #actual = self.session.run(q)
            #op=p.assign(p)
            ops.append(p.assign(q))
            
        self.session.run(ops)
        
        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})
        
    def fit(self,states,targets,actions):
        # call optimizer
        actions=np.atleast_1d(actions)
        targets=np.atleast_1d(targets)
        
        #scale target
        targets=(targets-self.min_y)/(self.max_y-self.min_y)

        states=np.atleast_2d(states)
        
        self.session.run(self.train_op,feed_dict={self.X: states,
            self.G: targets,self.actions: actions}         )
    
    
    def compute_diversity_of_selected_states(self):
        
        # first, compute the kernel matrix
        X=self.selected_exp
        Euc_dist=euclidean_distances(X,X)
        lengthscale=2
        
        KK=np.exp(-lengthscale*Euc_dist)
        temp=np.zeros(KK.shape)
        np.fill_diagonal(temp,0.01)
        KK=KK+temp
   
        #Wi, LW, LWi, W_logdet = pdinv(KK)
        #sign,W_logdet2=np.linalg.slogdet(KK)
        chol  = spla.cholesky(KK, lower=True)
        W_logdet=np.sum(np.log(np.diag(chol)))
            
        return W_logdet
        # second, compute the log determinant
       
    def fit_prioritized_exp_replay(self):
        if self.count_exp < self.min_experiences:
            # don't do anything if we don't have enough experience
            return
        
        # Obtain random mini-batch from memory
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_sz)
                
        #print(tree_idx)
 
        states=[each[0][0] for each in batch]
        actions=[each[0][1] for each in batch]
        rewards=[each[0][2] for each in batch]
        next_states=[each[0][3] for each in batch]
        dones=[each[0][4] for each in batch]


        next_actions = np.argmax(self.predict(next_states), axis=1)
        targetQ=self.predict(next_states)
        next_Q=[q[a] for a,q in zip(next_actions,targetQ)]

        targets = [r + self.gamma*next_q 
           if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        #scale target
        #targets=[ (val-self.min_y)/(self.max_y-self.min_y) for val in targets]
        
        # call optimizer
        
        #print(targets)
        _,absolute_errors=self.session.run( [self.train_op,self.absolute_errors], 
                         feed_dict={self.X:states, 
                                self.G:targets, self.actions: actions,
                                self.ISWeights_:ISWeights_mb })
    
        # Update priority
        #print(absolute_errors)

        self.memory.batch_update(tree_idx, absolute_errors)
    

    def add_experience(self, s,a,r,s2,done):
       # Add experience to memory
        experience = s, a, r, s2, done
        self.memory.store(experience)
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
          return np.random.choice(self.K)
        else:
          X = np.atleast_2d(x)
          return np.argmax(self.predict(X)[0])

def test_output_range(env, min_y,max_y):
    
    observation=env.reset()
    done=False
    totalreward=0
    iters=0
    K = env.action_space.n

    
    reward=0

    while not done and iters<2000:
        
        action=np.random.choice(K)
        
        observation, reward, done, infor= env.step(action)
        
        if done:
            reward=-200
            
        totalreward += reward



        if totalreward<min_y:
            min_y=totalreward
        if totalreward>max_y:
            max_y=totalreward

    return min_y,max_y

        
def play_one_episode_dueling_dqn_cartpole_per(env, qmodel,gamma,eps):
    
    observation=env.reset()
    done=False
    totalreward=0
    iters=0
    
    reward=0
    
    while not done and iters<2000:
        
        qmodel.count_exp+=1
        action=qmodel.sample_action(observation,eps)
        
        prev_obs=observation
        observation, reward, done, infor= env.step(action)
        
        totalreward += reward

        if done:
            reward=-200

        # update the model
        qmodel.add_experience(prev_obs,action,reward,observation,done)
        qmodel.fit_prioritized_exp_replay()
        
        #env.render()
        iters+=1


    return totalreward

        
def play_one_episode_dueling_dqn_cartpole_exp_replay(env, qmodel,gamma,eps):
    
    observation=env.reset()
    done=False
    totalreward=0
    iters=0
    
    reward=0
    
    while not done and iters<2000:
        
        action=qmodel.sample_action(observation,eps)
        
        prev_obs=observation
        observation, reward, done, infor= env.step(action)
        
        totalreward += reward

        if done:
            reward=-200

        # update the model
        qmodel.add_experience(prev_obs,action,reward,observation,done)
        qmodel.fit_exp_replay()
        
        #env.render()
        iters+=1


    return totalreward

def play_one_episode_dueling_dqn_cartpole(env, qmodel, gamma,eps):
    
    observation=env.reset()
    done=False
    totalreward=0
    iters=0
    
    reward=0
    
    while not done and iters<2000:
        
        action=qmodel.sample_action(observation,eps)
        
        prev_obs=observation
        observation, reward, done, infor= env.step(action)
        
        totalreward += reward

        if done:
            reward=-200

        Qnext=qmodel.predict(observation)
        
        G=reward+gamma*np.max(Qnext)

        qmodel.fit(prev_obs,G,action)
                
        iters+=1

    return totalreward

def evaluate_dueling_dqn_with_maxiter(x):
    
    gamma,lr,MaxIter=x


    tf.reset_default_graph()


    env=gym.make('CartPole-v0')
    #env=gym.make('MountainCar-v0')

    D = env.observation_space.shape[0]
    K = env.action_space.n
    
    #print("state_space:",D," action_space:",K)
        

    
    #pmodel = PolicyModel(D, K, [5])
    qmodel = DQNModel(D, K,  hidden_layer_sizes = [50,50], gamma=gamma,lr=lr)
    

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    qmodel.set_session(session)
    

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        #print(filename)
        
        #monitor_dir = filename + "_" + str(datetime.now())
        monitor_dir = filename

        env = wrappers.Monitor(env, monitor_dir)
        
    N = np.int(MaxIter)
    totalrewards = np.empty(N)
    #costs = np.empty(N)

    #print(min_y,max_y)
    
    for n in range(N):
        eps = 2.0/np.sqrt(n+1)

        #totalrewards[n]= play_one_episode_dueling_dqn_cartpole(env,qmodel,  gamma,eps)
        #totalrewards[n]= play_one_episode_dueling_dqn_cartpole_exp_replay(env,qmodel,  gamma,eps)
        totalrewards[n]= play_one_episode_dueling_dqn_cartpole_per(env,qmodel,  gamma,eps)


    #print("ave reward for last 100 episodes:", totalrewards[-100:].mean())
    #print("ave reward for last 100 episodes:", totalrewards.sum())

    #ave_reward=[np.mean(totalrewards[max(0,idx-100):idx+1] ) for idx in range(len(totalrewards))]
       
    #return totalrewards[max(0,n-100):n+1].mean()        
    return totalrewards      

        
            
#if __name__ == '__main__':
    #sys.argv = ["monitor"]

    #main()       
        