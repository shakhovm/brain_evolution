import gym

env = gym.make('CartPole-v1')

env.render()
env.reset()
for i in range(1000):
    env.step(env.action_space.sample())