from sys import path
import numpy as np
import torch
from torch._C import device
import torch.optim as optim
from collections import deque, namedtuple
import time

from gym_utils import query_environment
from hparams import HyperParams as hp
from model import Actor, Critic
from running_state import ZFilter
from ReplayBuffer import ReplayBuffer
from utils import get_action, save_checkpoint, update_model, eval
from annealing import anneal_sets

from torch.utils.tensorboard import SummaryWriter


def rollout(env_name, log_dir, num_agents = 5, num_iteration = 3000, kernel="Gaussian",  
            algorithm="SVPG", anneal_method = "cyclical", update_frequency = 1,
            render= False):
    """
    Inputs:
        -env_name :
        -policy_distribution : used distribution to model policy
        -num_epoches : number of epoches used for training
        -tempature : hyperparameter for exploration in variational RL
        -num_agents : number of agents training in parallel
        -kernel : used kernel in SVGD
                  ["Gaussian", "IMQ", "Matern"]
        -prior_dist : prior distribution for regularize the policy
                      ["uniform","gaussian","beta"]
        -algorithm : used algorithm
                     ["PG", "SVPG"]
    -------------------------------------------------------------------
        -rewards : list of rewards of each epoch
        -steps : list of steps of each epoch
    """
    #Environments
    env, env_info = query_environment(env_name)
    env.seed(17)

    # #Action Rescale
    # if env.action_space is None:
    #     action_bias = torch.tensor(0.)
    #     action_scale = torch.tensor(1.)
    # else:
    #     action_bias = torch.FloatTensor((env.action_space.high-env.action_space.low) /2. )
    #     action_scale = torch.FloatTensor((env.action_space.high+env.action_space.low) /2. )

    action_dim = env_info["Action Space"].shape[0]
    obs_dim = env_info["Observation Space"].shape[0]
    envs = [env for _ in range(num_agents)]


    #Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device : {}".format(torch.cuda.get_device_name(device)))

    #Agents
    actors = [Actor(obs_dim, action_dim).to(device) for _ in range(num_agents)]
    critics = [Critic(obs_dim).to(device) for _ in range(num_agents)]

    actor_optims = [optim.Adam(actors[i].parameters(), lr=hp.actor_lr) for i in range(num_agents)]
    critic_optims = [optim.Adam(critics[i].parameters(), lr=hp.critic_lr) for i in range(num_agents)] #weight_decay=hp.l2_rate
    
    #Algorithms
    if algorithm == "PG":
        from Agents.PG import train_model

    elif algorithm == "SVPG":
        from Agents.SVPG import train_model
        
        if kernel == "Gaussian":
            from SVPG_utils import Gaussian_Kxx_dxKxx
            kernel = Gaussian_Kxx_dxKxx
        elif kernel == "IMQ":
            from SVPG_utils import IMQ_Kxx_dxKxx
            kernel = IMQ_Kxx_dxKxx
        elif kernel == "Matern":
            from SVPG_utils import Matern_Kxx_dxKxx
            kernel = Matern_Kxx_dxKxx



    #Annealing_method:
    if anneal_method == "linear":
        annealer = anneal_sets(num_iteration, update_frequency, anneal_method=anneal_method)
    elif anneal_method == "tanh":
        annealer = anneal_sets(num_iteration, update_frequency, anneal_method=anneal_method)
    elif anneal_method == "cyclical":
        annealer = anneal_sets(num_iteration, update_frequency, anneal_method=anneal_method)
    else:
        raise ValueError("Unknown Annealing Method. Should be one of [cyclical,tanh,linear]")

    #Others
    np.random.seed(17)
    running_state = ZFilter((obs_dim,), clip=5)


    #comment = algorithm + '_' + kernel + '_' +anneal_method + '_' + env_name
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    #Running
    num_updates = 0
    episodes = 0
    for iter in range(num_iteration):
        score_list = []
        memory = [deque() for j in range(num_agents)]
        for i in range(num_agents):
            actors[i].eval(), critics[i].eval()

            steps = 0
            scores = []
            while steps < 2048:
                episodes += 1
                state = envs[i].reset()
                state = running_state(state)
                score = 0
                for _ in range(10000):
                    steps += 1
                    mu, std, _ = actors[i](torch.Tensor(state).to(device).unsqueeze(0))
                    action = get_action(mu, std)[0]
                    next_state, reward, done, _ = env.step(action)
                    next_state = running_state(next_state)

                    if done:
                        mask = 0
                    else:
                        mask = 1

                    memory[i].append([state, action, reward, mask])

                    score += reward
                    state = next_state

                    if done:
                        break

                scores.append(score)
            score_avg = np.mean(scores)
            score_list.append(score_avg)
        score_epi = np.max(score_list)

        if iter%update_frequency == 0:
            for j in range(num_agents):
                actors[j].train(), critics[j].train()
            #Anneal Temperature
            temperature = annealer.annealing(num_updates)
            num_updates += 1
            train_model(actors, critics, memory, actor_optims, critic_optims, num_agents=num_agents, kernel=kernel, temperature=temperature)
        
        print('{} episode score is {:.2f}, temperature {} iter {}'.format(episodes, score_epi,temperature,iter))
        writer.add_scalar('log/score', float(score_epi), iter)












                





