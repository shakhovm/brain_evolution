from NEAT.model import Phenotype
from NEAT.genotype import Genotype

import torch
import gym
from random import uniform
from math import pi
from time import sleep

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
    
        val = model.forward(torch.FloatTensor( data ).unsqueeze(0) )
        val = val.detach().numpy()
        print(val)
        val = classify(val)
    
        data = env.step(val)[0]
        angle = data[2]
        pos = data[0]
        if not is_alive(angle, pos): break
    env.close()

if __name__ == '__main__':
    MODE = 'vizualise'
    
    gt = Genotype(4,1,1)
    for i in range(4):
        gt.add_edge(i, 4, uniform(0, 1))
    gt.add_edge(4, 5, uniform(0, 1))
    
    for _ in range(10):
        r = uniform(0,1)
        if r < 0.1: gt.add_rand_node()
        elif 0.1 <= r <= 0.4: gt.add_rand_edge()
        elif 0.4 < r < 0.8: gt.mutate_rand_edge()
        else: gt.mutate_rand_weight()
    
    for x in gt.get_nodes():print(x)
    for x in gt.get_edges():print(x)
    
    if MODE == 'train':
        iters = 100
        half_width = 2.4
        score = 0
        model = Phenotype(gt)
        best_score = 0

        env = gym.make('CartPole-v0')
        while iters:
            score = 0
            env.reset()
            iter_data = env.step(0)[0]
            for _ in range(1000):
                # env.render()
                # sleep(1/60)

                val = model(torch.FloatTensor( iter_data ).unsqueeze(0))
                val = val.detach().numpy()
                val = classify(val)

                iter_data = env.step(val)[0]
                iter_angle = iter_data[2]
                iter_pos = iter_data[0]
                
                if not is_alive(iter_angle, iter_pos):
                    if score > best_score:
                        best_score = score
                        best_weights = model.state_dict()
                    
                    r = uniform(0,1)
                    if r < 0.1: gt.add_rand_node()
                    elif 0.1 <= r <= 0.4: gt.add_rand_edge()
                    elif 0.4 < r < 0.8: gt.mutate_rand_edge()
                    else: gt.mutate_rand_weight()
                        
                    model = Phenotype(gt)
                    
                    print("best score: {}; {} iters left".format(best_score, iters))
                    break
                score += 1
            env.close()
            iters -= 1

        print(best_score)
        torch.save(model.state_dict(), 'model')
        for x in gt.get_nodes():print(x)
        for x in gt.get_edges():print(x)
    elif MODE == 'vizualise':
        model = Phenotype(gt)
#         model.load_state_dict(torch.load('model'))
        demo(model)
    