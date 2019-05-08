import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.autograd as autograd 
import torch.nn.functional as F
import pickle 

from lib.model import * 
from lib.zfilter import ZFilter
from lib.util import *
from lib.trpo import trpo_step
from lib.data import *
import scipy.optimize

def trpo_learn(args):
    #env params
    env_name, batch_size, vv, als, ex_path, fig_path = args.env_id, args.batch_size, args.vv, args.als, args.ex_path, args.fig_path
    #trpo params
    max_kl, cr_lr, cg_step_size, damping = args.max_kl, args.cr_lr, args.cg_step_size, args.damping
    #data 
    data_n_steps, max_genert_num, gamma, lambd = args.data_n_steps, args.max_genert_num, args.gamma, args.lambd

    #set up 
    env = gym.make(env_name)
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    zflt = ZFilter((env.observation_space.shape[0],), clip=5)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    #model and optim
    policy_model =ModelActor(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
    print(env.observation_space.shape[0])
    critic_model =ModelCritic(env.observation_space.shape[0]).to(device)
    #opt_policy = optim.Adam(policy_model.parameters(), lr = args.lr_policy)
    opt_critic = optim.Adam(critic_model.parameters(), lr = args.lr_critic)

    # data generate 
    gene = generate(policy_model, env, env_name, als, device, data_n_steps, ex_path, fig_path, vv, max_genert_num, zflt)

    #train ... 
    V_loss, P_loss = [], []
    for trj in gene: 
        states, actions, rewards, dones =trj['states'],trj['actions'],trj['rewards'],trj['dones']
        states = torch.from_numpy(np.stack(states)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(actions)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(rewards)).to(dtype).to(device)
        dones = torch.from_numpy(np.stack(dones)).to(dtype).to(device)
        with torch.no_grad():
            values = critic_model(states)
            old_logprob = policy_model.get_log_prob(states, actions)
        adv, ref = cal_adv_ref(rewards, dones, values, gamma, lambd, device)

        opt_iter = int(math.ceil(states.shape[0]/batch_size))
        V_loss_, P_loss_ = [], []
        #for epoch in range(args.ppo_epoches):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)
        #states, actions, ref = states[perm].clone(), actions[perm].clone(), ref[perm].clone()
        #adv, old_logprob = adv[perm].clone(), old_logprob[perm].clone()
        """update critic, another way to optimize, which uses bfgs"""
        v_loss = 0
        '''
        def get_value_loss(flat_params):
            set_params(critic_model, torch.tensor(flat_params))
            for param in critic_model.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            values_pred = critic_model(states)
            value_loss = (values_pred - ref).pow(2).mean()
            print(values_pred)
            print(ref)
            # weight decay
            for param in critic_model.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            value_loss.backward()
            v_loss = value_loss.data.cpu().numpy()
            print(v_loss)
            return value_loss.item(), get_flat_grad_from(critic_model.parameters()).cpu().numpy()

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,get_params(critic_model).detach().cpu().numpy(), maxiter=25)
        set_params(critic_model, torch.tensor(flat_params))

        '''
        #critic optim
        for i in range(10):
            opt_critic.zero_grad()
            values = critic_model(states)
            loss_v = F.mse_loss(values, ref)
            loss_v.backward()
            v_loss = loss_v.data.cpu().numpy()
            opt_critic.step()
    
            #print(v_loss)
        #actor optim
        def get_loss():
            log_prob = policy_model.get_log_prob(states, actions)
            action_loss_v = -adv* torch.exp(log_prob - old_logprob)
            return action_loss_v.mean()
        def get_kl():
            return policy_model.get_kl(states, policy_model)

        p_loss = trpo_step(policy_model, get_loss, get_kl, max_kl, cr_lr, cg_step_size, damping, device)
        P_loss.append(p_loss)
        V_loss.append(v_loss)
    pickle.dump((policy_model, critic_model, zflt), open(ex_path+env_name+'_model_'+als+vv+'.p', 'wb'))    
    plot(0, V_loss, fig_path+'/loss/', env_name+als+vv+'v_loss')
    plot(1, P_loss, fig_path+'/loss/', env_name+als+vv+'p_loss')

