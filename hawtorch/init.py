import torch
import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if 'Conv' in classname:
        init.normal(m.weight.data, 0.0, 0.02)
    elif 'Linear' in classname:
        init.normal(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if 'Conv' in classname:
        init.xavier_normal(m.weight.data, gain=1)
    elif 'Linear' in classname:
        init.xavier_normal(m.weight.data, gain=1)
    elif 'BatchNorm' in classname:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if 'Conv' in classname:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif 'Linear' in classname:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif 'BatchNorm' in classname:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if 'Conv' in classname:
        init.orthogonal(m.weight.data, gain=1)
    elif 'Linear' in classname:
        init.orthogonal(m.weight.data, gain=1)
    elif 'BatchNorm' in classname:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)