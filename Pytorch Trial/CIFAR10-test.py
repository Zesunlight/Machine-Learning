import os
import torch as t
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, init_weights=True):
        super(Net, self).__init__()

        # input n*3*32*32
        # batch channel height width
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 10)

        if init_weights:
            self._initialize_weights()

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

    def _initialize_weights(self):
        para = r'A:\DataSet\Parameters\cifar10_para.pkl'
        if os.path.exists(para):
            self.load_state_dict(t.load(para))
            print('load parameters')
        else:
            for m in self.modules():
                if isinstance(m, t.nn.Linear):
                    t.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        t.nn.init.constant_(m.bias, 0)


def data_load():
    transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    trainset = tv.datasets.CIFAR10(root=r'A:\DataSet\Cifar10', transform=transform)
    train_loader = t.utils.data.DataLoader(
                        trainset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=8)

    testset = tv.datasets.CIFAR10(root=r'A:\DataSet\Cifar10', transform=transform)
    test_loader = t.utils.data.DataLoader(
                        testset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=8)

    return train_loader, test_loader


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    accuracy, train_loss = 0, 0

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    trainloader, testloader = data_load()

    for epoch in range(3):
        correct, total = 0, 0
        for n, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)

            _, predict = t.max(outputs, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (n + 1) % 256 == 0:
                print("epoch-{}, batch-{:>5} loss: {:.5f}".format(epoch, n + 1, train_loss / (256 * 32)))
                train_loss = 0
        print(f'epoch-{epoch} train accuracy: {correct} / {total} = {correct / total}')

        correct, total = 0, 0
        with t.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predict = t.max(outputs, 1)
                correct += (predict == labels).sum().item()
                total += labels.size(0)

        print(f'epoch-{epoch} test accuracy: {correct} / {total} = {correct / total}')
        if correct / total > accuracy:
            accuracy = correct / total
            t.save(net.state_dict(), r'A:\DataSet\Parameters\cifar10_para.pkl')
            print(f'save {accuracy} accuracy')
