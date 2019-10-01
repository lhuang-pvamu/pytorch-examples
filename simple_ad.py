import torch
import numpy as np
from torch.autograd import Variable

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

w = Variable(torch.Tensor([1.0]), requires_grad=True)
b = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return w*x + b

def loss(y_h,y):
    return (y_h - y)**2

print("predict before training", 4, forward(4).data[0])

for epoch in range(1000):
    for x_val, y_val in zip(x_data, y_data):
        y_h = forward(x_val)
        l = loss(y_h, y_val)
        l.backward()
        #print("grad: ", x_val, y_val, w.data, b.data, w.grad.data[0], b.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        b.data = b.data - 0.01 * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

    #print("progress: ", epoch, l.data[0])

print("predict after training", 4, forward(4).data[0])

