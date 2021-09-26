import torch
import math
import numpy as np
from torch.distributions import Normal
from hparams import HyperParams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_action(mu, std, explore=True):
    if explore:

        #print(mu)
        action = torch.normal(mu,std)
        #action = action.to('cpu')
        action = torch.tanh(action).to('cpu')
        #print(action)
        #action = action * action_scale + action_bias

        return action.data.numpy()
    else:
        action = mu.to("cpu")
        return action.data.numpy()

def log_density(x, mu, std, logstd, epsilon = 1e-6):
    
    normal = Normal(mu, std)
    y = torch.tanh(x)
    log_prob = normal.log_prob(x)
    # Enforcing Action Bound
    log_prob -= torch.log((1 - y.pow(2)))
    log_prob = log_prob.sum(1, keepdim=True)

    #var = std.pow(2)
    #log_density = -(x - mu).pow(2) / (2 * var) \
    #              - 0.5 * math.log(2 * math.pi) - logstd 

    return log_prob
    
    

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)




def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Parameters on different types!')
    return old_param_device



def parameters_to_vector(parameters, grad=False, both=False):
    param_device = None
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.view(-1))

        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                print(param)
                print(param.grad)
                vec.append(param.grad.detach().view(-1))
        return torch.cat(vec)

    else:
        param_vec = []
        grad_vec = []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            param_vec.append(param.view(-1))
            grad_vec.append(param.grad.detach().view(-1))
        return torch.cat(param_vec), torch.cat(grad_vec)


def vector_to_parameters(vector, parameters, grad=True):
    param_device = None
    pointer = 0

    if grad:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param
    else:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vector[pointer: pointer + num_param].view(param.size())
            pointer += num_param


def eval(env, actor, critic, running_state, max_episode_length, 
            action_low, action_high, action_bias, action_scale):
    score = 0
    actor.eval(), critic.eval()
    state = env.reset()
    #state = running_state(state)
    step = 0
    
    while step < max_episode_length:

        step += 1
        mu, std, _ = actor(torch.Tensor(state).to(device).unsqueeze(0))

        action = get_action(mu, std, action_bias, 
                            action_scale, explore=False)[0]
        action = np.clip(action, action_low, action_high)

        next_state, reward, done, _ = env.step(action)
        #next_state = running_state(next_state)

        state = next_state
        score += reward
            
        if done:
            break

    return score

def soft_update(critic, critic_target):
    for param,target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(target_param*(1-hp.tau)+param*hp.tau)

def hard_update(critic, critic_target):
    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
        target_param.data.copy_(param)