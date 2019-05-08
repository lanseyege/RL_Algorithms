import torch
import torch.nn.functional as F

def ppo_step(policy_model, critic_model, opt_critic, opt_policy, states, actions, ref, adv, old_logprob, ppo_eps=0.2, l2_reg=1e-3, qnew=None):
    #critic
    if critic_model is not None:
        values = critic_model(states)
        #print(values.size())
        loss_value = (values - ref).pow(2).mean()

        for param in critic_model.parameters():
            loss_value += param.pow(2).sum() * l2_reg
        V_loss = loss_value.data.cpu().numpy()
        opt_critic.zero_grad()
        loss_value.backward()
        opt_critic.step()
    
    #policy
    #mu = policy_model(states)
    #logprob = cal_log_prob(mu, policy_model.logstd, actions)
    logprob = policy_model.get_log_prob(states, actions)
    #print(np.shape(logprob.data.cpu().numpy()))
    ratio = torch.exp(logprob - old_logprob)
    #print(np.shape(ratio.data.cpu().numpy()))
    surr_obj = ratio * adv
    #print(np.shape(surr_obj.data.cpu().numpy()))
    clip_surr_obj = torch.clamp(ratio, 1.0-ppo_eps, 1.0+ppo_eps) * adv
    #print(np.shape(clip_surr_obj.data.cpu().numpy()))
    loss_policy = -torch.min(surr_obj, clip_surr_obj).mean()
    #print(np.shape(loss_policy.data.cpu().numpy()))
    opt_policy.zero_grad()
    P_loss = loss_policy.data.cpu().numpy()
    loss_policy.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 40)
    opt_policy.step()

    if critic_model is not None:
        return V_loss, P_loss
    return -1, P_loss


