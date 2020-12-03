import torch
import torch.nn as nn
from DDPG.updating_model import UpdatingModel


class ActorModel(nn.Module, UpdatingModel):
    def __init__(self, alpha, tau, input, output):
        super(ActorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        )
        self.tau = tau
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        # print(x)
        out = self.model(x)
        return torch.tanh(out)



