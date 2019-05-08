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

vv = '5'
mm = '9'
seed = 0
t = 6
use_cuda = True
ex_path = './data/expert/'
envs = ['Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2']
env_name = envs[t]
path = './data/expert/'+env_name+'_DModel_vv'+vv+'mm'+mm+'ac2'
env = gym.make(env_name)
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
#zflt = ZFilter((env.observation_space.shape[0],), clip=5)
dtype = torch.float64
torch.set_default_dtype(dtype)
 
experts = np.load(ex_path+env_name+'_ppo'+vv+'_state_action.npy')

ex_states_actions_ = experts[np.random.randint(0,experts.shape[0], 2000), :]
l1 = env.observation_space.shape[0] 
l2 = env.action_space.shape[0]

model = ModelDCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
model.load_state_dict(torch.load(path))
#torch.load(path)
model.eval()
A = experts.shape
deltas = []
d = np.random.randint(0,experts.shape[0], 10)
inx = 0
for a in d:#range(1):
    print('inx %d' %inx)
    inx += 1
    state = experts[a][:l1]
    delta = 0
    print(a)
    b = np.random.randint(0,experts.shape[0], 500)
    for i in b:#range(A[0]):
        #print(i)
        c = np.random.randint(0,experts.shape[0], 500)
        for j in c:#range(A[0]):
            in1 = np.hstack([state, experts[i][l1:]])
            in2 = np.hstack([state, experts[j][l1:]])
            in1 = torch.from_numpy(in1).to(device)
            in2 = torch.from_numpy(in2).to(device)
            de = (model(in1) - model(in2))**2
            #de = (model(in1) - model(in2))
            delta += de.data.cpu().numpy()
    print(delta)
    deltas.append(delta)
print(deltas)
print(np.mean(deltas))

