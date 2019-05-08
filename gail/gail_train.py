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

def gail_learn(args):
    '''env params'''
    env_name, batch_size, vv, mm, als, ex_path, fig_path = args.env_id, args.batch_size, args.vv,args.mm, args.als, args.ex_path, args.fig_path
    '''ppo params'''
    ppo_eps, ppo_epoches = args.ppo_eps, args.ppo_epoches
    '''data '''
    data_n_steps, max_genert_num, gamma, lambd = args.data_n_steps, args.max_genert_num, args.gamma, args.lambd

    '''set up '''
    env = gym.make(env_name)
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    #zflt = ZFilter((env.observation_space.shape[0],), clip=5)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    '''model and optim'''
    policy_model =ModelActor(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
    print(env.observation_space.shape[0])
    #if args.action == '3':
    #    critic_model =ModelCritic(2*env.observation_space.shape[0]).to(device)
    #else:
    critic_model =ModelCritic(env.observation_space.shape[0]).to(device)
    opt_policy = optim.Adam(policy_model.parameters(), lr = args.lr_policy)
    opt_critic = optim.Adam(critic_model.parameters(), lr = args.lr_critic)
    '''
        args.action == '0' : standard GAIL
        args.action == '1' : GAIL without expert action
        args.action == '2' : GAIL without expert action, but input agent action
    '''
    if args.action == '1':
        D = ModelDCritic(env.observation_space.shape[0], 0).to(device)
    else:
        D = ModelDCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    opt_D = optim.Adam(D.parameters(), lr = args.lr_critic)
    if args.action == '2':
        pass

    zflt = pickle.load(open(ex_path+env_name+'_expert'+vv+'.p', 'rb'))
    gene = generate2(policy_model, env, env_name, als, device, data_n_steps, ex_path, fig_path, vv, max_genert_num, zflt, critic_model, args.action, args.seed, expert_reward, D, mm)
    d_criterion = nn.BCELoss()
    experts = np.load(ex_path+env_name+'_ppo'+vv+'_state_action.npy')
    ex_states_actions_ = experts#[np.random.randint(0,experts.shape[0], ),:]

    E_loss, G_loss, V_loss, P_loss = [], [], [], []
    L_idx = 0
    for trj in gene:
        L_idx += 1
        states, actions, rewards, dones, values = trj['states'], trj['actions'], trj['rewards'], trj['dones'], trj['values']
        states = torch.from_numpy(np.stack(states)).to(dtype).to(device)
        actions = torch.from_numpy(np.stack(actions)).to(dtype).to(device)
        rewards = torch.from_numpy(np.stack(rewards)).to(dtype).to(device)
        dones = torch.from_numpy(np.stack(dones)).to(dtype).to(device)
        values = torch.from_numpy(np.stack(values)).to(dtype).to(device).unsqueeze(-1)
        with torch.no_grad():
            #values = critic_model(states)
            #values = expert_reward(D, states, actions)
            old_logprob = policy_model.get_log_prob(states, actions)
        adv, ref = cal_adv_ref(rewards, dones, values, gamma, lambd, device)
        ''' discrim optimization '''
        for _ in (range(1)):
            if args.action == '1':
                t = env.observation_space.shape[0]
                ex_states_actions_ = experts[np.random.randint(0,experts.shape[0], 2000), :t]
                ge_q_value = D(states)
            elif args.action == '0':
                ex_states_actions_ = experts[np.random.randint(0,experts.shape[0], 2000),:]
                ge_q_value = D(torch.cat([states, actions], 1))
            elif args.action == '2':
                t = env.observation_space.shape[0]
                rd = np.random.randint(0,experts.shape[0], 2000)
                ex_states_actions_ = experts[rd, :t]
                ex_states = torch.tensor(experts[rd,:t]).unsqueeze(0).to(device)
                with torch.no_grad():
                    ex_actions = policy_model.select_action(ex_states)[0].cpu().numpy()
                ge_q_value = D(torch.cat([states, actions], 1))
                ex_states_actions_ = np.hstack([ex_states_actions_, ex_actions])
            ex_states_actions = torch.from_numpy(ex_states_actions_).to(device)
            ## 1A train on real/expert
            ex_q_value = D(ex_states_actions)
            opt_D.zero_grad()
            loss_ex = d_criterion(ex_q_value,torch.zeros((ex_q_value.shape[0],1), device=device))
            E_loss.append(loss_ex.data.cpu().numpy())
            #print(loss_ex.data.cpu().numpy())
            ## 1B train on fake/generate
            loss_ge = d_criterion(ge_q_value, torch.ones((ge_q_value.shape[0],1), device=device))
            G_loss.append(loss_ge.data.cpu().numpy())
            loss_d = loss_ex + loss_ge
            loss_d.backward()
            opt_D.step()

        opt_iter = int(math.ceil(states.shape[0]/batch_size))
        P_loss_ = []
        V_loss_ = []
        for epoch in range(args.ppo_epoches):
            perm = np.arange(states.shape[0])
            #np.random.shuffle(perm)
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
                #print(b_states.size())
                #print(b_actions.size())
                #print(b_ref.size())
                #print(ref.size())
                #qnew = expert_reward(D, b_states, b_actions, args.action)
                #b_ref = qnew
                v_loss, p_loss = ppo_step(policy_model, critic_model, opt_critic, opt_policy, b_states, b_actions, b_ref, b_adv, b_old_logprob)
                P_loss_.append(p_loss)
                V_loss_.append(v_loss)
        P_loss.append(np.mean(P_loss_))
        V_loss.append(np.mean(V_loss_))
    signs = ''
    if args.action == '1':
        signs = '_no_action'
        pp = fig_path+'loss_ac1/'
    elif args.action == '2':
        signs = '_ag_action'
        pp = fig_path+'loss_ac2/'
    else:
        signs = ''
        pp = fig_path+'loss/'
    signs += '_seed'+str(args.seed)
    plot(0, E_loss, pp, env_name+als+'vv'+vv+'mm'+mm+'E_loss'+signs)
    plot(1, G_loss, pp, env_name+als+'vv'+vv+'mm'+mm+'G_loss'+signs)
    plot(2, V_loss, pp, env_name+als+'vv'+vv+'mm'+mm+'V_loss'+signs)
    plot(3, P_loss, pp, env_name+als+'vv'+vv+'mm'+mm+'P_loss'+signs)




