import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import numpy as np
import torch
from hparams import HyperParams as hp
from utils import log_density, soft_update, get_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_gae(rewards, masks, values):
    returns = torch.zeros_like(rewards).to(device)
    advants = torch.zeros_like(rewards).to(device)

    running_returns = 0
    previous_value = 0
    running_advants = 0
    with torch.no_grad():
        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                        values.data[t]
            running_advants = running_tderror + hp.gamma * hp.lamda * \
                            running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, actions):
    mu, std, logstd = actor(states)
    log_policy = log_density(actions, mu, std, logstd)
    advants = advants.unsqueeze(1)

    surrogate = advants * log_policy
    return  -surrogate.mean()


def train_critic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = states[batch_index]
            target = returns.unsqueeze(1)[batch_index]

            v_1,v_2 = critic(inputs)

            loss_1 = criterion(v_1, target)
            loss_2 = criterion(v_2, target)
            loss = loss_1 + loss_2

            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()


def train_actor(actor, advants, states, actions, actor_optim):
    loss = surrogate_loss(actor, advants, states, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, critic, critic_target, memory, actor_optim, critic_optim, 
                action_scale, action_bias):

    memory = np.array(memory, dtype=object)
    
    index = np.random.choice(memory.shape[0], hp.batch_size, replace=False)
    memory = memory[index]
    states = np.vstack(memory[:, 0])
    states = torch.Tensor(states).to(device)

    next_states = np.vstack(memory[:, 3])
    next_states = torch.Tensor(next_states).to(device)
    actions = torch.Tensor(list(memory[:, 1])).to(device).squeeze(1)

    rewards = torch.Tensor(list(memory[:, 2])).to(device)
    masks = torch.Tensor(list(memory[:, 4])).to(device)
    

    with torch.no_grad():
        # Select action according to policy
        mu, std, log_std = actor(next_states)
        next_action = torch.Tensor(get_action(mu,std,action_scale,action_bias)).cuda()
        #next_log_prob = log_density(next_action, mu, std, log_std).squeeze(1)

        # Compute the next Q values: min over all critics targets
        next_q_values_1, next_q_values_2 = critic_target(next_states, next_action)
        
        next_q_values = torch.min(next_q_values_1, next_q_values_2).squeeze(1)
        
        # add entropy term
        # next_q_values = next_q_values - hp.ent_coef * next_log_prob
        # td error + entropy term
        target_q_values = rewards + (1 - masks) * hp.gamma * next_q_values


    # Get current Q-values estimates for each critic network
    # using action from the replay buffer
    current_q_values_1, current_q_values_2 = critic(states, actions)

    actor.train(), critic.train()
    # Compute critic loss
    criterion = torch.nn.MSELoss()
    critic_loss = (criterion(current_q_values_1.squeeze(1), target_q_values).sum() + \
                            criterion(current_q_values_2.squeeze(1), target_q_values).sum())

    # Optimize the critic
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    #Update Target Network
    soft_update(critic, critic_target)


    # Train Actor
        
    online_mu, online_std, online_log_std = actor(states)
    online_action = get_action(online_mu, online_std, action_scale, action_bias)
    online_action = torch.Tensor(online_action).to(device)
    log_prob = log_density(online_action, online_mu, online_std, online_log_std)

    
    current_q_values_1, current_q_values_2 = critic(states, online_action)
    current_q_values = torch.min(current_q_values_1,current_q_values_2)

    agent_policy_grad = -log_prob*(rewards + hp.gamma * next_q_values * masks - current_q_values)
    actor_optim.zero_grad()
    policy_loss = agent_policy_grad.mean()
    policy_loss.backward()
    actor_optim.step()
    
    