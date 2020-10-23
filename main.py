import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import gym
from time import sleep

class Brain(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Brain, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()     
        
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
        
def classify(v):
    return 1 if v > 0.5 else 0

if __name__ == '__main__':
    brain = Brain(4,2)
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        sleep(1/30)
        
        val = brain.forward(torch.FloatTensor(np.array([1,2,3,1])))
        val = val.detach().numpy()
        val = classify(val)
        
        env.step(val)
    env.close()
    