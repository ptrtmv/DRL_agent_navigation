'''
Created on May 2, 2019

@author: ptrtmv
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNetwork(nn.Module):
    
    def __init__(self,stateSize,actionSize, hiddenLayers, dropProb = .0, seed = 0): 
        """
        Create a torch network with densely connected layers
        Args:
            stateSize (int):
                The dimension of the state space. (number of input features)
            actionSize (int):
                The number of possible discrete features. (number of nodes in 
                the output layer)
            hiddenLayers (list(int)):
                List with the sizes of the hidden layers (exp. [64,32])
            dropProb (double):
                Drop probability for the drop-out regularization. 
            seed (int):
                The seed for the random generator used in "torch.manual_seed(seed)"
        """
        
        super(DenseNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.inputSize = stateSize
        self.outputSize = actionSize
        
        hiddenSizes = zip(hiddenLayers[:-1],hiddenLayers[1:])
        
        # initialize the layers
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(self.inputSize,hiddenLayers[0])]) #input layer
        self.layers.extend([nn.Linear(ni,ni1) for ni,ni1 in hiddenSizes]) # hidden layers
        self.output = nn.Linear(hiddenLayers[-1],self.outputSize) # output layer
        
        self.dropout = nn.Dropout(p=dropProb)
        
        self.lossCriterion = nn.MSELoss()
        
    def forward(self,state):
        """
        Forward propagation through the network. 
            state -> Q(state,action)
        Args:
            state: 
                input state; input features 
        """        
        nextA = state
        for layer in self.layers:
            nextA = F.relu(layer(nextA))
            nextA = self.dropout(nextA)
        return self.output(nextA)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        