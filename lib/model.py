import torch
import torch.nn as nn
from lib.util import normal_log_density

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size, active='tanh', hidden_size=128, lstd=-0.0):
        super(ModelActor, self).__init__()
        if active == 'tanh':
            self.active = torch.tanh
        else:
            pass
        self.linear1 = nn.Linear(obs_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, act_size)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

        self.logstd = nn.Parameter(torch.ones(1, act_size) * lstd)

    def forward(self, x):
        x = self.linear1(x)
        x = self.active(x)
        x = self.linear2(x)
        x = self.active(x)
        mean = self.linear3(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        return mean, logstd, std

    def select_action(self, x):
        mean, _, std = self.forward(x)
        #action = mean + std * torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(std))
        action = torch.normal(mean, std)
        #print(mean)
        #print(std)
        #print(action)
        #print(torch.normal(mean, std))
        #torch.normal(mean, std)
        return action, mean, std

    def get_log_prob(self, x, action):
        mean, logstd, std = self.forward(x)
        return normal_log_density(action, mean, logstd, std)

    def get_kl(self, x, model):
        mean, logstd, std = self.forward(x)
        #mean_, logstd_, std_ = model(x)
        mean_, logstd_, std_ = mean.detach(), logstd.detach(), std.detach()
        kl = logstd - logstd_ + (std_ ** 2 + (mean_ - mean) ** 2) / (2.0 * std ** 2) - 0.5
        return kl.sum(1, keepdim=True)

    def get_ent(self, x ):
        mean, logstd, std = self.forward(x)
        return (logstd + 0.5 * np.log(2.0*np.pi*np.e)).sum(-1)

class ModelCritic(nn.Module):
    def __init__(self, obs_size, hidden_size = 128):
        super(ModelCritic, self).__init__()

        self.active = torch.tanh
        self.linear1 = nn.Linear(obs_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.active(x)
        x - self.linear2(x)
        x - self.active(x)
        value = self.linear3(x)
        return value

class ModelDCritic(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=128, ):
        super(ModelDCritic, self).__init__()
 
        self.active = torch.tanh
        self.linear1 = nn.Linear(obs_size+act_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)


    def forward(self, x):
        x = self.linear1(x)
        x = self.active(x)        
        x = self.linear2(x)
        x = self.active(x)
        value = torch.sigmoid(self.linear3(x))

        return value


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, active='tanh', hidden_size=128):
        super(ModelActor, self).__init__()
        self.active = torch.tanh
        self.linear1 = nn.Linear(obs_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, act_size)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)


    def forward(self, x):
        x = self.linear1(x)
        x = self.active(x)
        x = self.linear2(x)
        x = self.active(x)
        x = self.linear3(x)
        return self.active(x)

    def select_action(self, x):
        x = self.forward(x)

    def get_log_prob(self, ):
        pass
    def get_kl(self, ):
        pass
    def get_ent(self,):
        pass

class DDPGCritic(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, ):
        pass
