import gym
from DDPG.ddpg_agent import DDPGAgent
from parameters.pendulum_params import PARAMS

env = gym.make('Pendulum-v0')
agent = DDPGAgent(action_n=1,state_n=3,**PARAMS)
# agent.train(env)
agent.play(env)
# env.close()