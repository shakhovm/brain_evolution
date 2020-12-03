import torch
import torch.nn as nn
import torch.nn.functional as F
from DDPG.updating_model import UpdatingModel


class CriticModel(nn.Module, UpdatingModel):
    def __init__(self, alpha, tau, input, output, action_range):
        super(CriticModel, self).__init__()
        self.action_range = action_range
        self.state_layer = nn.Linear(input, 64)
        self.action_layer = nn.Linear(output, 64)
        self.shared_layers = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output)
        )
        self.alpha = alpha
        self.tau = tau
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, action):
        state_layer = F.relu(self.state_layer(state))
        action_layer = F.relu(self.action_layer(action))
        shared = torch.cat([state_layer, action_layer], dim=1)
        out = self.shared_layers(shared)
        return torch.tanh(out) * self.action_range
