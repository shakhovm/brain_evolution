import gym
from DDPG.ddpg_agent import DDPGAgent
from parameters.box_params import PARAMS

env = gym.make('BipedalWalker-v3')
# print(env._n_action_space)
# env.reset()
#
agent = DDPGAgent(action_n=4, state_n=24, **PARAMS)
agent.train(env)
# agent.play(env)
