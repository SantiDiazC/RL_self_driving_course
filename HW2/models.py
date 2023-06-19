import torch
import collections
import random
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class ReplayBuffer():
    def __init__(self, config={"buffer_limit": 5000}):
        self.buffer = collections.deque(maxlen=config["buffer_limit"])  # .deque allows append and discard at both ends

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # Randomly extract n(batch_size = 32) data from self.buffer
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])# adding dimension to make all same dimension (n=2)
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst), dtype=torch.int64), \
            torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_prime_lst), dtype=torch.float),\
            torch.tensor(np.array(done_mask_lst))

    def size(self):
        return len(self.buffer)

