import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse
from tensorboardX import SummaryWriter
from lib.model import * 

envs = ['Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2']

writer = SummaryWriter()
#set up 
env_name = envs[0]
env = gym.make(env_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

policy_model =ModelActor(env.observation_space.shape[0],env.action_space.shape[0]).to(device)

critic_model =ModelCritic(env.observation_space.shape[0]).to(device)
opt_policy = optim.Adam(policy_model.parameters(), lr = 0.01)
opt_critic = optim.Adam(critic_model.parameters(), lr = 0.01)

for name, param in policy_model.named_parameters():
    writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
writer.close()

