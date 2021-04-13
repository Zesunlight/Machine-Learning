import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # LeNet-5
    def __init__(self):
        super(Net, self).__init__()

        # input n*3*32*32
        # batch channel height width
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        # print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = self.fc3(x)

        return x

net = Net()
# print(net)
# for name,parameters in net.named_parameters():
#     print(name,':',parameters.size())
input = t.randn(2, 3, 32, 32)
out = net(input)

target = t.arange(2)
optimizer = t.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

loss = criterion(out, target)
net.zero_grad()
loss.backward()
optimizer.step()

