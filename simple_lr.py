#
# a simple logistic regression code using Pytorch
#

import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        # instantiate a nn.Linear module
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1,1) # one input and one output
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

# use MSE Loss
criterion = torch.nn.MSELoss(size_average=False)
# use SGD optimizer
# model.parameters() returns the learnable parameters of the model
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters())
traced_model = torch.jit.trace(model,x_data)
print(traced_model.code)

#training loop
for epoch in range(10000):
    y_pred = traced_model(x_data)
    #y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    #print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_var = Variable(torch.Tensor([[4.0]]))
print("predict after training: ", 4.0, model.forward(test_var).item())


