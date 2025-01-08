import torch,time,os
import torch.nn as nn
import torch.nn.functional as F
import math
from shared_parameters  import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import random
import  matplotlib.pyplot as plt
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):
    '''
    Approaximation function of spike firing rate function
    这就是脉冲神经元模型中的一种激活方式，当输入大于0时，输出为 1，否则为 0。这种激活函数常用于脉冲神经网络
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) #ctx用于传播反向传播时需要的信息
        return input.gt(0.).float() #用于返回一个与 input 具有相同形状的张量，其中每个元素表示是否大于0（即 True 或 False）。gt 是“greater than”的缩写，转换为浮点0或1

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors #是包含前向传播保存的变量的上下文对象
        grad_input = grad_output.clone() #grad_output 是来自上一层的梯度，将上一层的损失赋值给grad_input
        temp = abs(input) < lens #这行代码生成一个布尔张量 temp，其元素表示 input 中的每个元素是否在某个范围内（由 lens 决定）
        return grad_input * temp.float() #temp.float() 将布尔值 True 和 False 转换为浮点数（1.0 和 0.0），因此，只有当 input 的值在某个范围内时，梯度才会被传递给 grad_input

# Network structure
cfg_fc = [512, 10] #全连接，第一层512 第二层10

probs = 0.0 # dropout rate
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    这个函数的目的是根据训练的轮次调整学习率。"""
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1 #每隔 lr_decay_epoch 个 epoch，学习率会按 0.1 的因子减少

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self ):
        super(SNN_Model, self).__init__()

        self.fc1 = nn.Linear(28*28, cfg_fc[0], )#将输入的大小（28x28，即 784）映射到 cfg_fc[0] 大小（即 512）
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )#fc1 的输出映射到 cfg_fc[1] 大小（即 10）


        self.alpha1 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-3* torch.rand(1,cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-3 * torch.rand(1,cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, 784)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)


    def mask_weight(self):
        #按掩码（mask1 和 mask2）修改全连接层的权重。
        self.fc1.weight.data = self.fc1.weight.data * self.mask1
        self.fc2.weight.data = self.fc2.weight.data * self.mask2

    def produce_hebb(self):
        #该方法生成 Hebbian 学习矩阵（通常用于突触加权更新）。它返回两个 Hebbian 矩阵 hebb1 和 hebb2。
        hebb1 = torch.zeros(784, cfg_fc[0], device=device)#存储第一个突触更新
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

    def parameter_split(self):
        '''
        Split the meta-local parameters and gp-based parameters for different update methods
        方法用于将模型的参数分为两个部分：基本参数（base parameters）和局部参数（local parameters）。这可能用于不同的更新策略
        '''
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param


    def forward(self, input,hebb1, hebb2, wins = time_window):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        #1_mem：表示第一层神经元的膜电位（membrane potential），通常是神经元在每次时间步更新时的状态。膜电位决定了神经元是否会产生动作电位（spike）。
        #h1_spike：表示第一层神经元的 脉冲活动，也就是神经元是否在某一时刻产生了脉冲（spike）。通常，如果膜电位达到某个阈值，神经元就会产生一个脉冲。
        #h1_sumspike：表示第一层神经元在整个时间窗口内的 累计脉冲，即从开始到当前时刻所有的脉冲数量的总和
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        # hebb1 = torch.zeros(784, cfg_fc[0], device=device)

        for step in range(wins):

            decay_factor = np.exp(- step / tau_w)

            x = input
            # x = input > torch.rand(input.size(), device = device) # generate spike trains
            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1= mem_update_nonplastic(self.fc1,  self.alpha1, self.beta1, self.gamma1,self.eta1,
                                          x*decay_factor, h1_spike, h1_mem, hebb1)

            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike, hebb2 = mem_update_nonplastic(self.fc2,  self.alpha2,  self.beta2, self.gamma2,self.eta2,
                                          h1_spike*decay_factor, h2_spike, h2_mem, hebb2)

            h2_sumspike = h2_sumspike + h2_spike

        outs = h2_mem/thresh

        return outs.clamp(max = 1.1), h1_sumspike, h2_sumspike, hebb1.data, hebb2.data, self.eta1, self.eta2


def mem_update_nonplastic(fc, alpha, beta, gamma, eta, inputs,  spike, mem, hebb):

    state = fc(inputs) #神经元的 状态
    mem = (mem - spike * thresh) * decay + state #当前膜电位减去已发放脉冲所对应的阈值（thresh），如果神经元有脉冲（spike=1），则膜电位会减少 thresh，否则膜电位保持不变
    #然后，膜电位会按照 decay 衰减，这模拟了神经元膜电位随时间衰减的现象。最后，加入当前输入状态（state），模拟神经元受到外部刺激的影响
    now_spike = act_fun(mem - thresh)
    # Update local modules
    # hebb = w_decay * hebb + torch.bmm((inputs * beta).unsqueeze(2), ((mem/thresh) - eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()
    # hebb = hebb.clamp(min = -4, max = 4)
    return mem, now_spike.float(), hebb #函数返回更新后的膜电位 mem，当前的脉冲信号 now_spike，以及 Hebbian 学习模块的状态 hebb


