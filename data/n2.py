import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy.stats

envs = ['Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'Ant-v2', 'Humanoid-v2', 'HalfCheetah-v2']
fpath = './npic3/'
t = 0
t = int(sys.argv[1])
vv=sys.argv[2]
mm=sys.argv[3]
#plt.show()

paths = '../expert/'+envs[t]+'_ppo'+vv+'_exp_rewards.npy'#'_plot.npy'#'_genv_rewards.npy'
Reward = np.load(paths)
#Data.append(data)
print(len(Reward))
print(len(Reward[1]))
#Reward = np.sum(Reward, axis=-1)
Rewards = []
for reward in Reward:
    Rewards.append(sum(reward))
print(len(Reward))
means = np.mean(Rewards, axis=-1)
stds = np.std(Rewards, axis=-1)
print(means)
print(stds)
#plt.plot(xran, [min(Rewards)]*200, '.')
#plt.plot(xran, [max(Rewards)]*200, '-')
#plot([means]*200, [means-stds*0.9]*200, [means+stds*0.9]*200, xran, 'g', 'Expert')
nums = 800
_means = [means]*nums
_low = [means-stds*0.9]*nums
_upper = [means+stds*0.9]*nums

def plot(data, low, upper, xran, color, label):
    plt.plot(xran, data, lw = 1, color=color, alpha=1)
    plt.fill_between(xran, low, upper, color=color, alpha=0.4)
    plt.plot(xran, _means, lw = 1, color='g', alpha=1)
    plt.fill_between(xran, _low, _upper, color='g', alpha=0.4)

    plt.legend(loc=4)
    plt.title(envs[t]+'_'+label)
    plt.xlabel('interactions: 10^x')
    plt.ylabel('episodic returns')
    plt.savefig(fpath+envs[t]+'_'+label+'.png')
    plt.close()

Data = []
for i in range(1,5):
    paths = '../expert/'+envs[t]+'_gailvv'+vv+'mm'+mm+'_seed'+str(i)+'_plot.npy'#'_genv_rewards.npy'
    data = np.load(paths)
    Data.append(data)
Data = np.stack(Data)
means = np.mean(Data, axis=0)
stds = np.std(Data, axis=0)
print(means.shape)
print(stds.shape)
k = nums*2048
xran = np.log10(np.arange(nums) * 2048)
#xran[0] = 0
plot(means, means-0.9*stds, means+0.9*stds, xran, 'r', 'GAIL')

Data2 = []
for i in range(1,5):
    paths = '../expert/'+envs[t]+'_gailvv'+vv+'mm'+mm+'_no_action_seed'+str(i)+'_plot.npy'#'_genv_rewards.npy'
    data = np.load(paths)
    Data2.append(data)
Data2 = np.stack(Data2)
means = np.mean(Data2, axis=0)
stds = np.std(Data2, axis=0)
print(means.shape)
print(stds.shape)
k = nums*2048
#xran = np.arange(200) * 2048
plot(means, means-0.9*stds, means+stds*0.9, xran, 'b', 'GAIL_no_action')


Data3 = []
for i in range(1,5):
    paths = '../expert/'+envs[t]+'_gailvv'+vv+'mm'+mm+'_agent_ac_seed'+str(i)+'_plot.npy'#'_genv_rewards.npy'
    data = np.load(paths)
    Data3.append(data)
Data3 = np.stack(Data3)
means = np.mean(Data3, axis=0)
stds = np.std(Data3, axis=0)
print(means.shape)
print(stds.shape)
k = nums*2048
#xran = np.log10(np.arange(200) * 2048)
plot(means, means-stds*0.9, means+stds*0.9, xran, 'y', 'GAIL_Agent_action')


Data4 = []
for i in range(1,5):
    paths = '../expert/'+envs[t]+'_gailvv'+vv+'mm'+mm+'_agent_st_seed'+str(i)+'_plot.npy'#'_genv_rewards.npy'
    data = np.load(paths)
    Data4.append(data)
Data4 = np.stack(Data4)
means = np.mean(Data4, axis=0)
stds = np.std(Data4, axis=0)
print(means.shape)
print(stds.shape)
k = nums*2048
#xran = np.arange(200) * 2048
plot(means, means-stds*0.9, means+stds*0.9, xran, 'k', 'GAIL_States')



