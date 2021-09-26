import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import numpy as np
import torch
from hparams import HyperParams as hp
from utils import log_density, soft_update, vector_to_parameters, parameters_to_vector, get_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_returns(rewards, masks):
    returns = torch.zeros_like(rewards).to(device)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns

def get_loss(actor, returns, states, actions, values):
    mu, std, logstd = actor(states)
    log_policy = log_density(actions, mu, std, logstd)
    returns = returns.unsqueeze(1)

    objective = (returns-values) * log_policy
    objective = objective.mean()
    return objective
'''
def get_gae(rewards, masks, values):
    returns = torch.zeros_like(rewards)#.to(device)
    advants = torch.zeros_like(rewards)#.to(device)
    running_returns = 0
    previous_value = 0
    running_advants = 0
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
    return  surrogate

'''
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

            values = critic(inputs)
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()




def train_model(actors, critics, memory, actor_optims, critic_optims, 
                            num_agents, temperature, kernel):
    
    policy_grads = []
    parameters = []
    
    for i in range(num_agents):
        memory_i = np.array(memory[i],dtype=object)
        
        states = np.vstack(memory_i[:, 0])
        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(list(memory_i[:, 1])).to(device)
        rewards = torch.Tensor(list(memory_i[:, 2])).to(device)
        masks = torch.Tensor(list(memory_i[:, 3])).to(device)
        with torch.no_grad():
            values = critics[i](states)
        

        returns = get_returns(rewards, masks)

        # Train Critic
        train_critic(critics[i],states,returns,critic_optims[i])


        # Train Actor
        agent_policy_grad = get_loss(actors[i],returns,states,actions,values)


        actor_optims[i].zero_grad()

        policy_grad = agent_policy_grad
        policy_grad.backward()
        param_vector, param_grad = parameters_to_vector(actors[i].parameters(), both=True)
        parameters.append(param_vector.unsqueeze(0))
        policy_grads.append(param_grad.unsqueeze(0))



    parameters = torch.cat(parameters)
    Kxx, dxKxx = kernel(parameters, num_agents)
    policy_grads =  (1./temperature) *torch.cat(policy_grads)

    #policy_grads = torch.cat(policy_grads)
    grad_logp = torch.mm(Kxx, policy_grads)
    grad_theta = -(grad_logp + dxKxx) / num_agents
    for i in range(num_agents):
        vector_to_parameters(grad_theta[i], actors[i].parameters(), grad=True)
        actor_optims[i].step()
    
