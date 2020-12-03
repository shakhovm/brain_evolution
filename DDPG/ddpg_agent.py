from DDPG.critic_model import CriticModel
from DDPG.actor_model import ActorModel
import gym
import torch
import torch.nn.functional as F
from DDPG.replay_buffer import ReplayBuffer
from stochastic.processes.noise import GaussianNoise
from DDPG.noise import OrnsteinUhlenbeckActionNoise

class DDPGAgent:
    def __init__(
            self,
            state_n,
            action_n,
            alpha_actor,
            alpha_critic,
            episodes,
            steps_per_episode,
            buffer_size,
            train_begin,
            batch_size,
            gamma,
            tau,
            epochs,
            episodes_to_print,
            episodes_to_save,
            path,
            reward_path,
            load_path_critic,
            load_path_actor,
            load_models,
            action_range
    ):

        # Model Creation
        self.critic_model = CriticModel(alpha_critic, tau, state_n, action_n, action_range)#.cuda()

        self.actor_model = ActorModel(alpha_actor, tau, state_n, action_n)#.cuda()
        if load_models:
            self.actor_model.load_state_dict(torch.load(load_path_actor))
            self.critic_model.load_state_dict(torch.load(load_path_critic))
        self.critic_model_target = CriticModel(alpha_critic, tau, state_n, action_n, action_range)  # .cuda()
        self.critic_model_target.load_state_dict(self.critic_model.state_dict())
        self.actor_model_target = ActorModel(alpha_actor, tau, state_n, action_n) #.cuda()
        self.actor_model_target.load_state_dict(self.actor_model.state_dict())
        # self.critic_model_target.to(torch.device('cuda:0'))
        # self.actor_model_target.to(torch.device('cuda:0'))
        # self.critic_model.to(torch.device('cuda:0'))
        # self.actor_model.to(torch.device('cuda:0'))

        # Params definitions
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.train_begin = train_begin
        self.batch_size = batch_size
        self.gamma = gamma
        self.epochs = epochs
        self.best_reward = -10e5

        # Output params
        self.episodes_to_print = episodes_to_print
        self.episodes_to_save = episodes_to_save
        self.path = path
        self.reward_path = reward_path

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Noise
        self.noise = OrnsteinUhlenbeckActionNoise(action_n)

    def train_models(self):
        for epoch in range(self.epochs):
            # Sampling and defining y and y hat
            state, action, reward, state_next, done = self.replay_buffer.sample(self.batch_size)
            y_hat = reward + self.gamma * self.critic_model_target(state_next,
                                                                   self.actor_model_target(state_next)) * (1 - done)

            y = self.critic_model(state, action)
            # Critic train
            critic_loss = F.smooth_l1_loss(y, y_hat.detach())
            self.critic_model.optimizer.zero_grad()
            critic_loss.backward()
            self.critic_model.optimizer.step()

            # Actor train
            actor_loss = -self.critic_model(state, self.actor_model(state)).mean()  # As we have gradient descent
            self.actor_model.optimizer.zero_grad()
            actor_loss.backward()
            self.actor_model.optimizer.step()

            # Update weights

            self.critic_model_target.update_weights(self.critic_model.parameters())
            self.actor_model_target.update_weights(self.actor_model.parameters())
        return actor_loss, critic_loss

    def train(self, env: gym.wrappers.time_limit.TimeLimit):
        score = 0
        for episode in range(self.episodes):

            state = env.reset()
            for step in range(self.steps_per_episode):
                action = self.actor_model(torch.from_numpy(state).float())

                action += self.noise.sample(self.batch_size)[0]
                state_next, reward, done, _ = env.step(action.detach().numpy())
                self.replay_buffer.push((state, action.detach().numpy(), reward / 100, state_next, done))
                score += reward
                if done:
                    break
                state = state_next

            actor_loss, critic_loss = 1000, 1000

            if len(self.replay_buffer) >= self.train_begin:
                actor_loss, critic_loss = self.train_models()

            if (episode + 1) % self.episodes_to_print == 0:
                print(f"For episode {episode + 1} score is {score / self.episodes_to_print}")
                print(f"Critic Loss is {actor_loss}, Actor Loss is {critic_loss}")
                if score > self.best_reward:
                    self.best_reward = score
                    self.save_model(self.path + '_best')

                score = 0

            if (episode + 1) % self.episodes_to_save == 0:
                self.save_model(self.path)

    def save_model(self, path):
        torch.save(self.actor_model_target.state_dict(), path + "_actor_model")
        torch.save(self.critic_model_target.state_dict(), path + "_critic_model")

    def play(self, env: gym.wrappers.time_limit.TimeLimit):
        state = env.reset()
        while True:
            action = self.actor_model(torch.from_numpy(state).float())
            action += self.noise.sample(self.batch_size)[0]
            state_next, reward, done, _ = env.step(action.detach().numpy())
            env.render()
            state = state_next
            if done:
                for i in range(100):
                    action = self.actor_model(torch.from_numpy(state).float())
                    action += self.noise.sample(self.batch_size)[0]
                    state_next, reward, done, _ = env.step(action.detach().numpy())
                    env.render()
                    state = state_next
                break
