import numpy as np
import math
import torch

def sample_random():
    pass

def to_device(device, *args):
    return [x.to(device) for x in args]


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def cal_adv_ref2(rewards, dones, values, gamma, lambd, device):
    adv, ref = [], []
    #values = args.critic_model(states).detach()
    pre_gae = 0.0
    pre_val = 0.0 

    for val, done, reward in zip(reversed(values), reversed(dones), reversed(rewards)):

        delta = reward + gamma*pre_val*(1-done)-val
        gae = delta + gamma*lambd*(1-done)*pre_gae
        pre_val = val
        pre_gae = gae
        adv.append(gae)
        ref.append(gae+val)
        
    adv = torch.FloatTensor(list(reversed(adv))).to(device)
    ref = torch.FloatTensor(list(reversed(ref))).to(device)
    adv = (adv-adv.mean())/adv.std()
    return adv, ref
 
def cal_adv_ref(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(device, rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    #print(type(values))
    #print(type(advantages))
    returns = values + advantages.to(device)
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns

def expert_reward(D, state, action, device, arg_action='0'):
    if arg_action == '1' or arg_action == '3':
        state_action = torch.tensor(state, dtype=torch.float64).to(device)
    else:
        state_action = torch.tensor(np.hstack([state, action]), dtype=torch.float64).to(device)
    with torch.no_grad():
        return -math.log(1 - D(state_action)[0].item())


def get_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


