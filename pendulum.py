import gym
from DDPG.agent.ddpg_agent import DDPGAgent
from DDPG.parameters.pendulum_params import PARAMS

env = gym.make('Pendulum-v0')
agent = DDPGAgent(action_n=1,state_n=3,**PARAMS)
# agent.train(env)
agent.play(env)