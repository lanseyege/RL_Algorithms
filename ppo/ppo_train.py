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
from lib.ppo import ppo_step
from lib.data import *

def ppo_learn(args):
    #env params
    env_name, batch_size, vv, als, ex_path, fig_path = args.env_id, args.batch_size, args.vv, args.als, args.ex_path, args.fig_path
    #ppo params
    ppo_eps, ppo_epoches = args.ppo_eps, args.ppo_epoches
    #data 
    data_n_steps, max_genert_num, gamma, lambd = args.data_n_steps, args.max_genert_num, args.gamma, args.lambd

    #set up 
    env = gym.make(env_name)
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    zflt = ZFilter((env.observation_space.shape[0],), clip=5)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    #model and optim
    policy_model =ModelActor(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
    print(env.observation_space.shape[0])
    critic_model =ModelCritic(env.observation_space.shape[0]).to(device)
    opt_policy = optim.Adam(policy_model.parameters(), lr = args.lr_policy)
    opt_critic = optim.Adam(critic_model.parameters(), lr = args.lr_critic)

    # data generate 
    gene = generate(policy_model, env, env_name, als, device, data_n_steps, ex_path, fig_path, vv, max_genert_num, zflt)

    #train ... 
    V_loss, P_loss = [], []
    for trj in gene: 
        _logstd = policy_model.logstd.data.cpu().numpy()
        print('policy model sigma:' )
        print(_logstd)
        states, actions, rewards, dones =trj['states'],trj['actions'],trj['rewards'],trj['dones']
        print(actions[-1])
        print(trj['mean'])
        print(trj['std'])
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
        for epoch in range(args.ppo_epoches):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            states, actions, ref = states[perm].clone(), actions[perm].clone(), ref[perm].clone()
            adv, old_logprob = adv[perm].clone(), old_logprob[perm].clone()
            for i in range(opt_iter):
                ind = slice(i * batch_size, min((i + 1) * batch_size, states.shape[0]))
                b_states = states[ind]
                b_actions = actions[ind]
                b_adv = adv[ind]
                b_ref = ref[ind]
                b_old_logprob = old_logprob[ind]
                v_loss, p_loss = ppo_step(policy_model, critic_model, opt_critic, opt_policy, b_states, b_actions, b_ref, b_adv, b_old_logprob)
                V_loss_.append(v_loss)
                P_loss_.append(p_loss)
        V_loss.append(np.mean(V_loss_))
        P_loss.append(np.mean(P_loss_))
    pickle.dump((policy_model, critic_model, zflt), open(ex_path+env_name+'_model_'+als+vv+'.p', 'wb'))    
    plot(0, V_loss, fig_path+'/loss/', env_name+als+vv+'v_loss')
    plot(1, P_loss, fig_path+'/loss/', env_name+als+vv+'p_loss')




