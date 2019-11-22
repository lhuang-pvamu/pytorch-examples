__author__ = 'Lei Huang'

# use auto differentiation to calcuate sin(x)
# 10/06/19


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from torch.autograd import Variable
import math
import random
import os
import argparse
import matplotlib.pyplot as plt

num_epoches = 1000

# calculate sin(x) using Tylor series
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

# test the gradient of sin(x) using PyTorch auto differentiation
def test_gradient_sin():
    x = Variable(torch.Tensor([1.0]), requires_grad=True)
    y = sin(x)
    print('y ==', y)

    y.backward()
    print('sin(x) gradient == ', x.grad.data)
    print('cos(x) ==', math.cos(1.0))

# build a neural network model for fitting sin(x)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128,128)
        self.fc7 = nn.Linear(128,128)
        self.fc8 = nn.Linear(128,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

def loss_func(output, target):
    criterion = nn.MSELoss()
    penalty = torch.sum(torch.abs(torch.where((output>=-1) & (output <=1), torch.zeros(1), output)))*0.000001
    loss = criterion(output, target) + penalty
    return loss

# train the model to fit sin(x)
def train():

    batch = 128
    net = Net()
    model_name = 'SineNet'
    summary(net, (1,1))


    criterion = nn.MSELoss() #L1Loss() # MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
    loss_hist = []


    for i in range(num_epoches):
        input = []
        result = []
        for j in range(batch):
            rand = random.uniform(0.0, 2.0 * math.pi)
            input.append(rand)
            result.append(sin(rand))

        x = torch.FloatTensor(input)
        x.unsqueeze_(-1)
        #print(x)
        output = net.forward(x)
        target = torch.FloatTensor(result)
        target.unsqueeze_(-1)
        #print(target)
        optimizer.zero_grad()
        loss = loss_func(output, target)
        #loss = criterion(output, target)
        loss_hist.append(loss)
        if i % 100 == 0:
            print(loss.item())
        if i < 1000 and i%10==0:
            plt.figure()
            plt.plot(input, output.data, 'o')
            plt.savefig('saved_figures/plot_{:03}.png'.format(i))
            plt.close()
        loss.backward()
        optimizer.step()

    return net, loss_hist

# calcuate sin(x)
def inference(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model_name = 'SineNet'
    if os.path.exists(os.path.join("./saved_models",model_name)):
        model.load_state_dict(torch.load(os.path.join("./saved_models",model_name),map_location=torch.device(device)))
        print("=== Load from a saved model:{0} ===".format(model_name))
    model.to(device)
    y = model.forward(torch.FloatTensor([x])).item()
    print('sin({})={}'.format(x, y), sin(x).item())
    return y


# use the trained model to find the inverse function of sin(x) using Adam optimization
def inverse(x):
    EPOCHS = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model_name = 'SineNet'
    if os.path.exists(os.path.join("./saved_models",model_name)):
        model.load_state_dict(torch.load(os.path.join("./saved_models",model_name),map_location=torch.device(device)))
        print("=== Load from a saved model:{0} ===".format(model_name))
    model.to(device)

    x0 = Variable(torch.FloatTensor([2.0]), requires_grad=True)
    target = torch.FloatTensor([x])
    criterion = nn.L1Loss() # MSELoss()
    optimizer = torch.optim.Adam([x0], lr=0.1, weight_decay=1e-5)

    for i in range(EPOCHS):
        output = model.forward(x0)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(x0.item())

    print('The inverse of sin(x)={}, x={}, model(x)={}.'.format(x, x0.item(), model.forward(x0).item()))
    return x0.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    training = True
    if training:
        model, loss_hist = train()
        torch.save(model.state_dict(), os.path.join("./saved_models/", "SineNet"))

        plt.figure()

        plt.title("Sin(x) Training Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Training Accuracy")
        plt.plot(range(1, num_epoches + 1), loss_hist, label="Training")
        #plt.ylim((0, 1.))
        plt.yscale("log")
        plt.xticks(np.arange(1, num_epoches + 1, 100.0))
        plt.legend()
        # plt.show()
        plt.savefig("./saved_figures/perf.png")


    y = inference(1.0)  # y=sin(x)
    x = inverse(y)      # x=sin^-1(y)


