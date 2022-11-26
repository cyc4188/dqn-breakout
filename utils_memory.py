from typing import (
    Tuple,
)

import torch
import pickle

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'states': self.__m_states,
                'actions': self.__m_actions,
                'rewards': self.__m_rewards,
                'dones': self.__m_dones,
                'capacity': self.__capacity,
                'pos': self.__pos,
                'size': self.__size,
            }, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.__m_states = data['states']
            self.__m_actions = data['actions']
            self.__m_rewards = data['rewards']
            self.__m_dones = data['dones']
            self.__capacity = data['capacity']
            self.__pos = data['pos']
            self.__size = data['size']

    def __len__(self) -> int:
        return self.__size
