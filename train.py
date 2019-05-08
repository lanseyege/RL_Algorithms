import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse

#from lib import *
from ppo.ppo_train import ppo_learn
from trpo.trpo_train import trpo_learn
from gail.save_ex import _save_expert
from gail.gail_train4 import gail_learn
#from trpo import *
#from gail import *

envs = ['Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2']

def train(args):
    '''
    als == 'ppo': run ppo
           'trpo': run trpo
           'gail': run gail
           'save': save expert trajectories
    '''
    print(args.als)
    if args.als == 'ppo':
        ppo_learn(args)
    elif args.als == 'trpo':
        trpo_learn(args)
    elif args.als == 'gail':
        gail_learn(args)
    elif args.als == 'save':
        _save_expert(args)

if __name__ == '__main__':
    ENV_ID = envs[0]
    #trpo param
    max_kl = 0.01
    cr_lr = 1e-3
    cg_step_size = 10
    damping = 0.1
    #ppo param
    ppo_eps = 0.2
    ppo_epoches = 10
    #gail param 
    max_genert_num = 200  #iterations
    max_expert_num = 50000 #timesteps
    #trajectory
    batch_size = 64
    data_n_steps = 2048
    ex_path = './data/expert/'
    fig_path = './data/figure/'
    #other 
    gamma = 0.99
    lambd = 0.95
    lr_policy = 3e-4
    lr_critic = 3e-4
    als = 'ppo' # default algorithm
    use_cuda = False
    parser = argparse.ArgumentParser(description='run RL algorithms...')
    parser.add_argument('--env_id', default=ENV_ID, help='enviroment name of Mujoco')
    parser.add_argument('--max_kl', default=max_kl, type=float, help='trpo max kl')
    parser.add_argument('--cr_lr', default=cr_lr, type=float, help='trpo cr_lr')
    parser.add_argument('--cg_step_size', default=cg_step_size,type=int,help='trpo cg step size')
    parser.add_argument('--damping', default=damping, type=float, help='trpo damping')
    parser.add_argument('--ppo_eps', default=ppo_eps, type=float, help='ppo eps')
    parser.add_argument('--ppo_epoches', default=ppo_epoches, type=int, help='ppo epoches')
    parser.add_argument('--max_genert_num', default=max_genert_num, type=int, help='max num for generating trajectories')
    parser.add_argument('--max_expert_num', default=max_expert_num, type=int, help='max num for generating expert trajectories')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='batch size ')
    parser.add_argument('--data_n_steps', default=data_n_steps, type=int, help='one trj length')
    parser.add_argument('--ex_path', default=ex_path, help='path to store expert data')
    parser.add_argument('--fig_path', default=fig_path, help='path to store picture')
    parser.add_argument('--gamma', default=gamma, type=float, help='param to calculate adv')
    parser.add_argument('--lambd', default=lambd, type=float, help='param to calculate adv')
    parser.add_argument('--lr_policy', default=lr_policy, type=float, help='learning rate')
    parser.add_argument('--lr_critic', default=lr_critic, type=float, help='learning rate')
    parser.add_argument('--als', default=als, help='algorithm you want to run: "ppo","trpo","gail","save" ')
    parser.add_argument('--vv', default='0', help='version, a sign of experinment')
    parser.add_argument('--mm', default='0', help='version, a sign of gail experinment')
    parser.add_argument('--action', default='0', help='0: with action; 1: without action in gail; 2: without action, but use agent action')
    parser.add_argument('--seed', default=1, type=int, help='random seed ')
    parser.add_argument('--use_cuda', default=use_cuda, type=bool, help='use cuda or not ')
    args = parser.parse_args()
    print(args)
    train(args)

