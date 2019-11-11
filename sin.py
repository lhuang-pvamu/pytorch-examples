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


def test_gradient_sin():
    x = Variable(torch.Tensor([1.0]), requires_grad=True)
    y = sin(x)
    print('y ==', y)

    y.backward()
    print('sin(x) gradient == ', x.grad.data)
    print('cos(x) ==', math.cos(1.0))


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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# train a model to fit sin(x)
def train():

    EPOCHS = 10000
    batch = 128
    net = Net()
    model_name = 'Net'
    summary(net, (1,1))


    criterion = nn.L1Loss() # MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)

    for i in range(EPOCHS):
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
        loss = criterion(output, target)
        if i % 100 == 0:
            print(loss.item())
        loss.backward()
        optimizer.step()

    return net

# use the trained model to find the inverse function of sin(x)
def inference(x):
    EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    model_name = 'Net'
    if os.path.exists(os.path.join("./saved_models",model_name)):
        model.load_state_dict(torch.load(os.path.join("./saved_models",model_name),map_location=torch.device(device)))
        print("=== Load from a saved model:{0} ===".format(model_name))
    model.to(device)

    x0 = Variable(torch.FloatTensor([3.14]), requires_grad=True)
    target = sin(x)
    criterion = nn.L1Loss() # MSELoss()
    optimizer = torch.optim.Adam([x0], lr=0.01, weight_decay=1e-5)

    for i in range(EPOCHS):
        output = model.forward(x0)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #print(x0.item())

    print(x0.item(),target,model.forward(torch.FloatTensor([x])), model.forward(x0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model = train()
    #torch.save(model.state_dict(), os.path.join("./saved_models/", "Net"))
    inference(2.0)

