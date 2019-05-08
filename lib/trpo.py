import torch
import torch.nn as nn
import numpy as np
from lib.util import get_params, set_params

def conjugate_gradients(Avp, b, device, nsteps, residual_tol=1e-10 ):
    '''
        calculate x in Hx=b using cg method. 
    '''
    x = torch.zeros(b.size()).to(device)
    r, p = b.clone(), b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, get_loss, x, delta, expect_, bt=10, ac_rt=0.1):
    fval = get_loss().data
    for (_, step) in enumerate(.5**np.arange(bt)):
        xnew = x + delta * step 
        set_params(model, xnew)
        fnew = get_loss().data
        act_imp = fval - fnew #- fval 
        exp_imp = expect_ * step
        ratio = act_imp / exp_imp
        if ratio.item() > ac_rt: #and act_imp.item() > 0:
            return True, xnew 
    return False, x

def trpo_step(model, get_loss, get_kl, max_kl, cr_lr, cg_step_size, damping, device):
    #get 1st order gradient of loss
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    '''
    get dot vector of 2nd order gradient of kl and conjugate vector p.  
    e.g. Ax = b, A is the 2nd gradient of kl, b is 1st 
        gradient of loss. This function is to get the 
        value A*p, p is a conjugate vector and we set 
        p=b before iteration. 
    '''
    def Fvp(v):
        ''' H*g, '''
        kl = get_kl()
        kl = kl.mean()
        '''
        1st order gradient of kl, set param True to construct 
        a graph to calculate higher order derivative products.
        '''
        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)

        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        '''
        2nd order gradient of kl 
            
        '''
        v_v = torch.tensor(v).to(device)
        kl_v = (flat_grad_kl * v_v).sum() # get H*v 
        grads = torch.autograd.grad(kl_v, model.parameters()) # get 2nd order gradient

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping # multiply damping to avoid zero??
    '''
        delta_ = sqrt(2*max_kl/(x*(H*x))), is a scalar
        delta  = delta_ * x is the estimated proppsed step 
        expect_ = (gradient of f(x) ) * (gradient of x)
    stepdir =  conjugate_gradients(Fvp, -loss_grad, device, 10)
    delta_ = torch.sqrt(2*max_kl/((stepdir*Fvp(stepdir)).sum(0, keepdim=True)))[0]#a scalara
    delta  = delta_ * stepdir
    # f'(x) * x' is used in line search as a throshold 
    expect_ = -delta_ * (loss_grad * stepdir).sum(0, keepdim=True) 
    old_params = get_params(model)
    sign, new_params = linesearch(model, get_loss, old_params, delta, expect_)
    set_params(model, new_params)
    return loss.data.cpu().numpy()  

    '''

    stepdir = conjugate_gradients(Fvp, -loss_grad, device,10)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs/max_kl)
    fullstep = stepdir/lm[0]
    neggdotstepdir = (-loss_grad*stepdir).sum(0, keepdim=True)

    old_params = get_params(model)
    ss, new_params = linesearch(model, get_loss, old_params, fullstep, neggdotstepdir/lm[0])
    set_params(model, new_params)
    return loss.data.cpu().numpy()
