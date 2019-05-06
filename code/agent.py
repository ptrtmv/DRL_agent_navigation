'''
Created on May 2, 2019

@author: ptrtmv
'''

import io

import numpy as np
import random
import pickle
from copy import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
from networks import DenseNetwork 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DrlAgent():
    """
    DRL Agent class
    """
    
    def __init__(self,brain):
        """
        Initialization of the agent object
        Args:
            brain:
                A Brain object controlling the learning and acting  
        """
                
        self.brain = brain     
        
    def act(self,state,eps):
        return self.brain.react(state,eps)
    
    def experience(self, state, action, reward, next_state, done):    
        self.brain.experience(state, action, reward, next_state, done)

    def saveTrainingStatus(self,pathToWeightsFile):        
        torch.save(self.brain.dqnLocal.state_dict(), pathToWeightsFile)


class DrlBrain():
    
    def __init__(self, stateSize, actionSize, 
                hiddenLayers = None, 
                gamma = 0.99,
                learningRate = 1e-4, 
                dropProb = .0,
                dqnUpdatePace = 4, 
                targetDqnUpdatePace = 1e-3,
                bufferSize = int(1e5),
                batchSize = 64, 
                batchEpochs = 1,
                 seed = None):
        """
        Args:
            stateSize:
                The dimension of the state space; number of features for the 
                deep Q-network
            actionSize:
                Number of possible actions; size of output layer
            hiddenLayers:
                List with sizes of output layers. (exp: [64,32])
            gamma:
                RL discount factor for future rewards (Bellman's return) 
            learningRate:
                The learning rate for the gradient descent while training the 
                (local) neural network; 
                corresponds more or less to the parameter alpha in
                RL controlling the how much the most recent episodes
                contribute to the update of the Q-Table                            
            dropProb:
                Drop probability for the drop-out regularization.
            dqnUpdatePace:
                Determines after how many state-action steps the local network 
                should  be updated. 
            targetDqnUpdatePace:
                If targetDqnUpdatePace < 1: a soft update is performed at each 
                local network update
                If targetDqnUpdatePace >= 1: the target network is replaced by 
                the local network
            bufferSize:
                Size of the memory buffer containing the experiences < s, a, r, s’ >
            batchSize:
                The batch size used in the gradient descent during learning
            batchEpochs:
                The number of epochs when training the network  
            
        """
        
        self.hiddenLayers = hiddenLayers
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.batchEpochs = batchEpochs
        
        self.targetDqnUpdatePace = targetDqnUpdatePace
        self.dqnUpdatePace = dqnUpdatePace
        self.numberExperiences = 0
        self.dropProb = dropProb
        self.learningRate = learningRate
        
        self.gamma = gamma
        
                
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.seed = seed
        
        if seed == None:
            self.seed = np.random.randint(0,999) 
        
        # Q-Network
        self.dqnLocal = DenseNetwork(self.stateSize, self.actionSize, self.hiddenLayers, self.dropProb, self.seed).to(device)
        self.dqnTarget = DenseNetwork(self.stateSize, self.actionSize, self.hiddenLayers, self.dropProb, self.seed).to(device)
        
        self.optimizer = optim.Adam(self.dqnLocal.parameters(), lr=self.learningRate)
        
        self.memory = Memory(self.bufferSize, self.batchSize, seed)
        
     
    
    def react(self,state,eps):
        """
        React to a state by returning the most probable action according to an
        epsilon-greedy policy.
         
        Args:
            state:
                The current state 
            eps:
                Epsilon value in the epsilon-greedy policy
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.dqnLocal.eval()
        with torch.no_grad():
            actions = self.dqnLocal(state)
        self.dqnLocal.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.actionSize))
    
    def getQvalues(self,state):
        """
        Get the Q(state,action) values for a given state.
        Args:
            state:
                the state which is to be propagated through the (local) network
        """
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.dqnLocal.eval()
        with torch.no_grad():
            actions = self.dqnLocal(state)
        self.dqnLocal.train()
        res = actions.cpu().data.numpy()
        return res
    
    
    def experience(self,state, action, reward, next_state, done):
        """
        Gather experiences < s, a, r, s’ > in the replay buffer and train the 
        deep Q-Network after dqnUpdatePace x Experience-Steps
        
        Args:
            state:
                current state
            action:
                current action
            reward:
                current reward
            next_state:
                next state resulting from the current action
            done:
                flag denoting if the episode is finished
        """
        self.numberExperiences += 1        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        if self.numberExperiences % self.dqnUpdatePace == 0:
            self._learn()
        
    
# # #     def getPickleReadyBrain(self):
# # #         pickledBrain = copy(self)
# # #         pickledBrain.dqnLocal = None
# # #         pickledBrain.dqnTarget = None
# # #         pickledBrain.optimizer = None
# # #         pickledBrain.memory = None
# # #         return pickledBrain
# # # 
# # #     @classmethod
# # #     def loadPickledBrain(cls,pathToPickledBrainFile,pathToWeigthsFile=None):
# # #         with open(pathToPickledBrainFile, 'rb') as input:
# # #             pickledBrain = pickle.load(input)
# # #             
# # #         # Q-Network
# # #         pickledBrain.dqnLocal = DenseNetwork(pickledBrain.stateSize, pickledBrain.actionSize, pickledBrain.hiddenLayers, pickledBrain.dropProb, pickledBrain.seed).to(device)
# # #         pickledBrain.dqnTarget = DenseNetwork(pickledBrain.stateSize, pickledBrain.actionSize, pickledBrain.hiddenLayers, pickledBrain.dropProb, pickledBrain.seed).to(device)
# # #         
# # #         pickledBrain.optimizer = optim.Adam(pickledBrain.dqnLocal.parameters(), lr=pickledBrain.learningRate)        
# # #         pickledBrain.memory = Memory(pickledBrain.bufferSize, pickledBrain.batchSize, pickledBrain.seed)
# # #         
# # #         if pathToWeigthsFile!=None:
# # #             pickledBrain.dqnLocal.load_state_dict(torch.load(pathToWeigthsFile))
# # #             pickledBrain.dqnTarget.load_state_dict(torch.load(pathToWeigthsFile))
# # #             
# # #         return pickledBrain
    
    def _learn(self):    
        """
        Train the (local) network and update the target network 
        """
        if len(self.memory) <= self.batchSize*self.batchEpochs:
            return
        
        for _ in range(self.batchEpochs):
            # repeat gradient descent for self.batchEpochs             
            self._batchStep(self.batchSize)
            
        # update the target network
        if self.targetDqnUpdatePace < 1:
            self.softTargetUpdate()
        elif self.numberExperiences %  (self.targetDqnUpdatePace) == 0: 
            self.hardTargetUpdate()
    
    
    def _batchStep(self,batchSize):
        """
        Perform a single batch gradient descent step with batch size of batchSize 
        """
        
        states, actions, rewards, nextStates, dones = self.memory.torchSample(batchSize)
        
        # Get max predicted Q values (for next states) from target model
        maxQ_dqnTarget = self.dqnTarget(nextStates).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * maxQ_dqnTarget * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.dqnLocal(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def softTargetUpdate(self):
        """
        Perform a soft update of the target parameters
        targetNetwork <- targetDqnUpdatePace * localNetwork + 
                            (1- targetDqnUpdatePace) * targetNetwork
        """
        for targetParam, localParam in zip(self.dqnTarget.parameters(), self.dqnLocal.parameters()):
            targetParam.data.copy_(self.targetDqnUpdatePace*localParam.data + (1.0-self.targetDqnUpdatePace)*targetParam.data)

     
    def hardTargetUpdate(self):
        """
        Perform a hard update of the target network
            targetNetwork <- localNetwork
        """        
        for targetParam, localParam in zip(self.dqnTarget.parameters(), self.dqnLocal.parameters()):
            targetParam.data.copy_(localParam.data)

        
        
        
        
        
        
        
 
class Memory():
    '''
    Memory buffer containing the experiences < s, a, r, s’ >
    '''    

    def __init__(self,bufferSize, batchSize, seed):
        '''  
        Initialize Memory object
        
        :param bufferSize (int): 
                maximum size of buffer
        :param batchSize (int): 
                size of each training batch
        :param seed (int): 
                random seed
        '''  
        self.memory = deque(maxlen=bufferSize)  
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
            
    
    def sample(self,batchSize):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batchSize)        
            
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        nextStates = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
  
        return (states, actions, rewards, nextStates, dones)    
    
    
    def torchSample(self,batchSize):
        states, actions, rewards, nextStates, dones = self.sample(batchSize)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        nextStates = torch.from_numpy(nextStates).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        return (states, actions, rewards, nextStates, dones) 
        
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        