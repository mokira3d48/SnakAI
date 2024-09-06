import os
import logging

import torch
import torch.nn.functional as F
from jedi.inference.arguments import repack_with_argument_clinic
from torch import nn
from torch import optim


LOG = logging.getLogger(__name__)


class LinearQNet(nn.Module):
    """
    :param input_size:
    :param hidden_size:
    :param output_size:

    :type input_size: `int`
    :type hidden_size: `int`
    :type output_size: `int`
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        :param x:
        :type x: `torch.Tensor`
        :rtype: `torch.Tensor`
        """
        x = self.linear1(x)
        x = F.relu(x)
        # x = self.linear2(x)
        # x = F.relu(x)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        :param file_name: The file name of the model. By default, the file
                          name will be named `model.pth`.
        :type file_name: `str`
        """
        model_folder_path = './outputs'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class QTrainer:
    """
    :param model:
    :param criterion:
    :param optimizer:
    :param gamma:

    :type model: `nn.Module`
    :type criterion: `nn.Module`
    :type optimizer: `optim.Optimizer`
    :type gamma: `float`
    """
    def __init__(self, model, criterion, optimizer, gamma):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.gamma = gamma

    @classmethod
    def get_new_instance(cls, model, gamma, lr, criterion=None):
        """
        :param model:
        :param gamma:
        :param lr:
        :param criterion:

        :type model: `nn.Module`
        :type gamma: `float`
        :type lr: `float`
        :type criterion: `nn.Module`|`None`
        """
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if not criterion:
            criterion = nn.MSELoss()
        instance = cls(model, criterion, optimizer, gamma)
        return instance

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)  # dim -> (1, x)
            next_state = torch.unsqueeze(next_state, 0)  # dim -> (1, x)
            action = torch.unsqueeze(action, 0)  # dim -> (1, x)
            reward = torch.unsqueeze(reward, 0)  # dim -> (1, x)
            done = (done, )

        # 1: Predict Q value using the current state.
        # 2: Compute Q_new = r + gamma * max(next_predicted Q value) * done
        #    with `done` value is 0 or 1. So, if done is equal to 0, q_new = r.
        prediction = self.model(state)
        target = prediction.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                neext_prediction = self.model(next_state[idx])
                q_new += self.gamma * torch.max(neext_prediction)
            action_id = torch.argmax(action).item()
            target[idx][action_id] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(prediction, target)
        loss.backward()
        self.optimizer.step()
