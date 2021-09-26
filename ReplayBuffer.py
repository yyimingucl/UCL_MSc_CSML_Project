from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen = size)
    
    def store(self, state, action, reward, done):
        state      = np.expand_dims(state, 0)
            
        self.buffer.append((state, action, reward, done))
    
    def sample(self, batch_size):
        state, action, reward, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, done
    
    def __len__(self):
        return len(self.buffer)