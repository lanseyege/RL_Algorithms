import pickle
import torch
import gym
from lib.data import save_expert

def _save_expert(args):
    als = 'ppo'
    vv = args.vv
    env_name = args.env_id
    env = gym.make(env_name)
    paths = args.ex_path
    #device=torch.device('cpu')
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    max_expert_num = args.max_expert_num 
    policy_model,_,zflt = pickle.load(open(paths+env_name+'_model_'+als+vv+'.p', 'rb'))
    save_expert(args, policy_model, env, env_name, als, device, _, paths, vv,max_expert_num ,zflt)
