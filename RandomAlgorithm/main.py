import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import gym
from math import pi
from time import sleep

class Brain(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Brain, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            torch.nn.init.xavier_uniform(self.fc1.weight)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            torch.nn.init.xavier_uniform(self.fc2.weight)
            self.sigmoid = torch.nn.Sigmoid()     
        
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output
        
def classify(v):
    return 1 if v > 0.5 else 0

def is_alive(angle, pos):
    return -0.5*pi < angle < 0.5*pi and abs(pos) < 2.0

def demo(model):
    env = gym.make('CartPole-v0')
    env.reset()
    data = env.step(0)[0]
    for _ in range(10000):
        env.render()
        sleep(1/30)
        
        val = model.forward(torch.FloatTensor( data ))
        val = val.detach().numpy()
        val = classify(val)
    
        data = env.step(val)[0]
        angle = data[2]
        pos = data[0]
        if not is_alive(angle, pos): break
    env.close()

if __name__ == '__main__':
    MODE = 'vizualise'
    
    if MODE == 'train':
        iters = 5000
        half_width = 2.4
        score = 0
        brain = Brain(4,2)
        best_weights = brain.state_dict()
        best_score = 0

        brain.load_state_dict(torch.load('./random_model/model_2'))

        env = gym.make('CartPole-v0')
        while iters:
            score = 0
            env.reset()
            # env.render()
            iter_data = env.step(0)[0]
            for _ in range(1000):
                # env.render()
                # sleep(1/60)

                val = brain.forward(torch.FloatTensor( iter_data ))
                val = val.detach().numpy()
                val = classify(val)

                iter_data = env.step(val)[0]
                iter_angle = iter_data[2]
                iter_pos = iter_data[0]
                
                if not is_alive(iter_angle, iter_pos):
                    if score > best_score:
                        best_score = score
                        best_weights = brain.state_dict()
                    brain = Brain(4,2)
                    print("best score: {}; {} iters left".format(best_score, iters))
                    break
                score += 1
            env.close()
            iters -= 1

        print(best_score)
        torch.save(brain.state_dict(), 'model')
    elif MODE == 'vizualise':
        brain = Brain(4,2)
        brain.load_state_dict(torch.load('./random_model/model_2'))
        demo(brain)
        