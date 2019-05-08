import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle 

from lib.util import to_device

def plot(frame_idx, rewards, fpaths, name):
    #plt.figure(figsize=(20,20))
    #plt.subplot(111)
    #plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.title(name+' (X-axis: iteration Y-axis: Reward(avg))')
    plt.plot(rewards)
    #plt.show()
    plt.savefig(fpaths+name+'.png')
    plt.close()


def generate(model, env, env_name, als, device, data_n_steps, paths, fpaths, vv, max_genert_num, zflt, nn='0', expert_reward=None, D=None):
    #device = torch.device('cpu')
    states, actions, rewards, dones, values = [], [], [], [], []

    k = 0
    genert_num = 0
    xran = []
    mean_rwds = []
    mean_drwds = []
    while genert_num < max_genert_num:
        genert_num += 1
        #for i in range(data_n_steps + 1):
        num_step = 0  
        num_episode = 0
        total_rwd = 0
        total_drwd = 0
        max_rwd = -1e6
        min_rwd = 1e6
        states, actions, rewards, dones, values = [], [], [], [], []
        while num_step < data_n_steps:        
            state = env.reset()
            state = zflt(state)
            rwd = 0.0
            drwd = 0.0
            for t in range(10000):
                k += 1
                #num_step += 1
                states.append(state)
                state_ = torch.tensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    #action_probs = model.select_action(state_)[0].cpu().numpy()
                    action_probs, mean, std = model.select_action(state_)
                    action_probs = action_probs[0].cpu().numpy()
                action_probs = action_probs.astype(np.float64)
                next_state, r, done, _ = env.step(action_probs)
                #next_state = np.clip(next_state, -5, 5)
                next_state = zflt(next_state)
                actions.append(action_probs)
                rwd += r
                if expert_reward is not None:
                    v = expert_reward(D, state, device, action_probs)
                    drwd += v
                    values.append(v)
                rewards.append(r)
                if done:
                    dones.append(0)
                else:
                    dones.append(1)

                if done:# or num_step >= data_n_steps:
                    #state = env.reset()
                    break
                state = next_state
            num_step += t + 1
            num_episode += 1
            total_rwd += rwd
            max_rwd = max(max_rwd, rwd)
            min_rwd = min(min_rwd, rwd)
            if expert_reward is not None:
                total_drwd += drwd
        yield {'states':states, 'actions':actions, 'rewards':rewards, 'dones':dones, 'values':values, 'mean':mean, 'std':std}
        xran.append(k)
        mean_rwd = total_rwd/num_episode
        mean_rwds.append(mean_rwd)
        print('ts %d\t genert_num %d\t min_rwd %.2f\t max_rwd %.2f\t mean_rwd %.2f\t' %(k,genert_num, min_rwd, max_rwd, mean_rwd))

    #plot(1, trj_rwds, 'halfcheetah-v2-5')
    plot(1, mean_rwds, fpaths, env_name+'_'+als+'vv'+vv)
    np.save(paths+env_name+'_'+als+vv+'_plot.npy', np.array(mean_rwds))
    np.save(paths+env_name+'_'+als+vv+'_ppo_rewards.npy', np.array(rewards))
    np.save(paths+env_name+'_'+als+vv+'_ppo_xran.npy', np.array(xran))

def save_expert(args, policy_model, env, env_name, als, device, data_n_steps, paths, vv, max_expert_num, zflt, mm='0'):
    num_steps = 0
    experts = []
    rewards = []
    #max_expert_num = 5000
    while num_steps < max_expert_num:
        state = env.reset()
        state = zflt(state)
        done = False
        reward = []
        while not done:      
            #state = np.clip(state, -5, 5)
            state_ = torch.tensor(state).unsqueeze(0).to(device)
            #action_probs = agent(state)
            with torch.no_grad():
                action_probs = policy_model.select_action(state_)[0].cpu().numpy()
            next_state, r, done, _ = env.step(action_probs)
            next_state = zflt(next_state)
            #actions.append(action_probs)
            reward.append(r)
            experts.append(np.hstack([state, action_probs]))
            state = next_state
            num_steps += 1
        rewards.append(reward)    
        print('num_steps % d reward %.4f ' %(num_steps, sum(reward)))
        

    experts = np.stack(experts)
    rewards = np.array(rewards)
    np.save(paths+env_name+'_'+als+vv+'_state_action.npy', experts)
    #np.save('./expert_trj/halfcheetah'+'_state_action5.npy', experts)
    np.save(paths+env_name+'_'+als+vv+'_exp_rewards.npy', rewards)
    #np.save('./expert_trj/halfcheetah'+'_rewards5.npy', rewards)
    pickle.dump(zflt, open(paths+env_name+'_expert'+vv+'.p', 'wb'))

def generate2(model, env, env_name, als, device, data_n_steps, paths, fpaths, vv, max_genert_num, zflt, critic_model, arg_action, seed, expert_reward=None, D=None, mm = '0'):
    #device = torch.device('cpu')
    states, actions, rewards, dones, values = [], [], [], [], []

    k = 0
    genert_num = 0
    xran = []
    mean_rwds = []
    mean_drwds = []
    rewards_env = []
    while genert_num < max_genert_num:
        genert_num += 1
        #for i in range(data_n_steps + 1):
        num_step = 0  
        num_episode = 0
        total_rwd = 0
        total_drwd = 0
        max_rwd = -1e6
        min_rwd = 1e6
        states, actions, rewards, dones, values = [], [], [], [], []
        while num_step < data_n_steps:        
            state = env.reset()
            state = zflt(state)
            rwd = 0.0
            drwd = 0.0
            prev_state_ = torch.tensor(state).unsqueeze(0).to(device)
            prev_state = state
            for t in range(10000):
                k += 1
                #num_step += 1
                states.append(state)
                state_ = torch.tensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    #action_probs = model.select_action(state_)[0].cpu().numpy()
                    action_probs, mean, std = model.select_action(state_)
                    action_probs = action_probs[0].cpu().numpy()
                    state__ = state_
                    #if arg_action == '3':
                    #    state__ = torch.cat([prev_state_, state_], -1)
                    value = critic_model(state__)[0][0].cpu().numpy()
                action_probs = action_probs.astype(np.float64)
                next_state, r, done, _ = env.step(action_probs)
                #next_state = np.clip(next_state, -5, 5)
                next_state = zflt(next_state)
                actions.append(action_probs)
                rwd += r
                #if expert_reward is not None:
                _state = state
                if arg_action == '3':
                    _state = np.hstack([prev_state, state])
                v = expert_reward(D, _state, action_probs, device, arg_action)
                drwd += v
                values.append(value)
                rewards_env.append(r)
                rewards.append(v)
                if done:
                    dones.append(0)
                else:
                    dones.append(1)

                if done:# or num_step >= data_n_steps:
                    #state = env.reset()
                    break
                prev_state_ = state_
                prev_state = state
                state = next_state
            num_step += t + 1
            num_episode += 1
            total_rwd += rwd
            max_rwd = max(max_rwd, rwd)
            min_rwd = min(min_rwd, rwd)
            if expert_reward is not None:
                total_drwd += drwd
        yield {'states':states, 'actions':actions, 'rewards':rewards, 'dones':dones, 'values':values, 'mean':mean, 'std':std}
        xran.append(k)
        mean_rwd = total_rwd / num_episode
        mean_drwd = total_drwd / num_episode
        mean_rwds.append(mean_rwd)
        mean_drwds.append(mean_drwd)
        print('ts %d\t genert_num %d\t dones %d\t min_rwd %.2f\t max_rwd %.2f\t mean_rwd %.2f\t mean_drwd %.2f\t' %(k, genert_num, sum(dones), min_rwd, max_rwd, mean_rwd, mean_drwd))

    #plot(1, trj_rwds, 'halfcheetah-v2-5')
    if als == 'gail':
        signs = ''
        if arg_action == '1':
            signs = '_no_action'
        elif arg_action == '2':
            signs = '_agent_ac'
        elif arg_action == '3':
            signs = '_agent_st'

        plot(1, mean_rwds, fpaths, env_name+'_'+als+'vv'+vv+'mm'+mm+signs+'_seed'+str(seed)+'_env_reward')
        plot(0, mean_drwds, fpaths, env_name+'_'+als+'vv'+vv+'mm'+mm+signs+'_seed'+str(seed)+'_define_reward')
        np.save(paths+env_name+'_'+als+'vv'+vv+'mm'+mm+signs+'_seed'+str(seed)+'_plot.npy', np.array(mean_rwds))
        np.save(paths+env_name+'_'+als+'vv'+vv+'mm'+mm+signs+'_seed'+str(seed)+'_genv_rewards.npy', np.array(rewards_env))
        np.save(paths+env_name+'_'+als+'vv'+vv+'mm'+mm+signs+'_seed'+str(seed)+'_xran.npy', np.array(xran))
    else:
        plot(1, mean_rwds, fpaths, env_name+'_'+als+vv)
        np.save(paths+env_name+'_'+als+vv+'_seed'+str(seed)+'_plot.npy', np.array(mean_rwds))
        np.save(paths+env_name+'_'+als+vv+'_seed'+str(seed)+'_xran.npy', np.array(xran))


