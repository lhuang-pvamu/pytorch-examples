__author__ = 'Lei Huang'

# use auto differentiation to calcuate sin(x)
# 10/06/19


import torch
import numpy as np
from torch.autograd import Variable
import math

def sin(x):
    sign = 1.0
    t = Variable(torch.Tensor([0.0]))

    for i in range(1, 20, 2):
        newterm = x**i / math.factorial(i)
        #print(i, newterm)
        t = t + sign * newterm
        #print(t)
        sign = - sign

    return t

x = Variable(torch.Tensor([1.0]), requires_grad=True)
y = sin(x)
print('y ==', y)

y.backward()
print('sin(x) gradient == ', x.grad.data)
print('cos(x) ==', math.cos(1.0))