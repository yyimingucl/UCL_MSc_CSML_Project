import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HyperParams as hp


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        
        self.mean = nn.Linear(hp.hidden, num_outputs)
        self.logstd = nn.Linear(hp.hidden, num_outputs)

        self.tanh = nn.Tanh()
        self.initialize()

        self.logstd_min = hp.logstd_min
        self.logstd_max = hp.logstd_max

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.mean.weight, gain=1)
        nn.init.constant_(self.mean.bias, 0)
        nn.init.xavier_uniform_(self.logstd.weight, gain=1)
        nn.init.constant_(self.logstd.bias, 0)
       

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))

        mu = self.mean(x)
        
        logstd = torch.clamp(self.logstd(x), self.logstd_min, self.logstd_max)

        std = torch.exp(logstd)

        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.tanh = nn.Tanh()
        self.initialize()
        
    
    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        nn.init.constant_(self.fc3.bias, 0)
        

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        v = self.fc3(x)
        return v
