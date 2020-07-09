# -*- coding: utf-8 -*-


#from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
# Inspired by https://github.com/dennybritz/reinforcement-learning

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from tqdm import tqdm

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

class PolicyModel:
    def __init__(self,D,K, lr,hidden_layer_sizes):
        # K: action_size
        # D: state_size
        
        self.layers=[]
        M1=D
        for M2 in hidden_layer_sizes:
            layer=HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1=M2
            
        #final layer
        final_layer=HiddenLayer(M1,K,tf.nn.softmax, use_bias=False)
        self.layers.append(final_layer)
        
        # inputs and targets
        self.X=tf.placeholder(tf.float32, shape=(None,D), name='X')
        self.actions=tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages=tf.placeholder(tf.float32,shape=(None,),name='advantages')
        
        # calculate output and cost
        Z=self.X
        for layer in self.layers:
            Z=layer.forward(Z)
            
        p_a_given_s=Z
        
        # action_scores=Z
        # p_a_given_s=tf.nn.softmax(action_scores)
        
        
        self.predict_op=p_a_given_s
        
        selected_probs=tf.log( tf.reduce_sum(p_a_given_s*tf.one_hot(self.actions,K),                                             
                                             reduction_indices=[1] ))
        
        cost= -tf.reduce_sum(self.advantages * selected_probs)
        
        self.train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        #self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
        #self.train_op = tf.train.GradientDescentOptimizer(1e-1).minimize(cost)
        
        
    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X=np.atleast_2d(X)
        actions=np.atleast_1d(actions)
        advantages=np.atleast_1d(advantages)
        
        self.session.run(self.train_op, feed_dict={self.X:X, 
                                                   self.actions:actions,
                                                   self.advantages:advantages})
    
    def predict(self,X):
        X=np.atleast_2d(X)
        return self.session.run(self.predict_op,feed_dict={self.X:X})
    
    def sample_action(self,X):
        p=self.predict(X)[0] # compute the prob
        return np.random.choice(len(p),p=p) # draw a random action
    
# approximates V(s)
class ValueModel:
    def __init__(self, D, lr,hidden_layer_sizes):
        # D: state_size
        
        self.layers=[]
        
        M1=D
        for M2 in hidden_layer_sizes:
            layer=HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1=M2
            
        #final layer
        final_layer=HiddenLayer(M1,1,lambda x:x)
        self.layers.append(final_layer)
        
        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        
        # calculate output and cost
        Z=self.X
        for layer in self.layers:
            Z=layer.forward(Z)
            
        Y_hat=tf.reshape(Z,[-1]) # output
        
        self.predict_op=Y_hat
        
        cost=tf.reduce_sum( tf.square(self.Y-Y_hat) )
        
        #self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    def set_session(self, session):
        self.session = session
        
    def partial_fit(self, X,Y):
        X=np.atleast_2d(X)
        Y=np.atleast_1d(Y)
        self.session.run(self.train_op, feed_dict={self.X:X,self.Y:Y})
        
    def predict(self,X):
        X=np.atleast_2d(X)
        return self.session.run(self.predict_op,feed_dict={self.X:X})
    
def play_one_episode_cartpole(env, pmodel, vmodel, gamma):
    
    observation=env.reset()
    done=False
    totalreward,iters=0,0
    
    reward=0
    
    while not done and iters<2000:
        
        action=pmodel.sample_action(observation)
        
        prev_obs=observation
        observation, reward, done, infor= env.step(action)
        
        totalreward += reward

        if done:
            reward=-200
            
        Vnext=vmodel.predict(observation)
        G=reward+gamma*Vnext
        vmodel.partial_fit(prev_obs,G)
        
        advantage=G-vmodel.predict(prev_obs)
        pmodel.partial_fit(prev_obs,action,advantage)
        iters+=1
    
    return totalreward, iters
        
def play_one_episode_cartpole_update_end(env, pmodel, vmodel, gamma):
    
    observation=env.reset()
    done=False
    totalreward=0
    iters=0
    
    states,actions,rewards=[],[],[]

    reward=0
    
    while not done and iters<2000:
        
        action=pmodel.sample_action(observation)
        
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        
        prev_obs=observation
        observation, reward, done, infor= env.step(action)
        
        #env.render()
        
        totalreward += reward

        if done:
            reward=-200
            
        iters+=1
        
    # save the final(s,a,r) tuple
    action=pmodel.sample_action(observation)
    states.append(observation)
    rewards.append(reward)
    actions.append(action)
    
    returns,advantages=[],[]
    G=0 # use the last reward, instead of 0
    
    for s,r in zip( reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G-vmodel.predict(s)[0])
        G=r+gamma*G
        
    returns.reverse()
    advantages.reverse()
    
    # update the models
    pmodel.partial_fit(states,actions,advantages)
    vmodel.partial_fit(states,returns)
    
    return totalreward, iters

def evaluate(x):
#def main():
    # x is the parameter
    # x[0]: gamma
    # x[1]: learning rate Policy model
    # x[2]: learning rate Value model
    
    #x=[0.99,0.0005,0.0005]
    
    gamma,lr_pm,lr_vm=x
    tf.reset_default_graph()

    env=gym.make('CartPole-v0')
    #env=gym.make('MountainCar-v0')

    D = env.observation_space.shape[0]
    K = env.action_space.n
    
    #print("state_space:",D," action_space:",K)
    
    pmodel = PolicyModel(D, K,lr_pm, [10])
    vmodel = ValueModel(D,lr_vm, [10])
        
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    
        
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        #print(filename)
        
        #monitor_dir = filename + "_" + str(datetime.now())
        monitor_dir = filename

        env = wrappers.Monitor(env, monitor_dir)
        
        
    N = 500
    totalrewards = np.empty(N)
        
    for n in range(N):
        totalrewards[n], num_steps= play_one_episode_cartpole_update_end(env, 
                    pmodel, vmodel, gamma)
        
        """
        if n%100==0:
            print("episode:", n, "total reward:", totalrewards[n],
                  "num_step:",num_steps,
            "aveg reward (last 100):", totalrewards[max(0,n-100):n+1].mean()) 
            """
    
    return totalrewards[max(0,n-100):n+1].mean()
    #print("ave reward for last 100 episodes:", totalrewards[-100:].mean())
    #print("ave reward for last 100 episodes:", totalrewards.sum())

    """
    ave_reward=[np.mean(totalrewards[max(0,idx-100):idx+1] ) for idx in range(len(totalrewards))]
       
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()  
    
    plt.plot(ave_reward)
    plt.title("Average Rewards")
    plt.show()   
    """
def evaluate_with_maxiter(x):
#def main():
    # x is the parameter
    # x[0]: gamma
    # x[1]: learning rate Policy model
    # x[2]: learning rate Value model
    
    #x=[0.99,0.0005,0.0005]
    
    gamma,lr_pm,lr_vm,MaxIter=x
    tf.reset_default_graph()

    env=gym.make('CartPole-v0')
    #env=gym.make('MountainCar-v0')

    D = env.observation_space.shape[0]
    K = env.action_space.n
    
    #print("state_space:",D," action_space:",K)
    
    pmodel = PolicyModel(D, K,lr_pm, [10])
    vmodel = ValueModel(D,lr_vm, [10])
        
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    
        
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        #print(filename)
        
        #monitor_dir = filename + "_" + str(datetime.now())
        monitor_dir = filename

        env = wrappers.Monitor(env, monitor_dir)
        
        
    N = int(MaxIter)
    totalrewards = np.empty(N)
        
    for n in range(N):
        totalrewards[n], num_steps= play_one_episode_cartpole_update_end(env, 
                    pmodel, vmodel, gamma)
        
        """
        if n%100==0:
            print("episode:", n, "total reward:", totalrewards[n],
                  "num_step:",num_steps,
            "aveg reward (last 100):", totalrewards[max(0,n-100):n+1].mean()) 
            """
    
    #return totalrewards[max(0,n-100):n+1].mean()        
    return totalrewards        

            
if __name__ == '__main__':
    #sys.argv = ["monitor"]

    main()       
        