import torch as t
import torch.nn as nn
import torchvision as tv
from experiment import Net

def data_load():
    transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ])

    trainset = tv.datasets.CIFAR10(root=r'A:\DataSet', transform=transform)

    train_loader = t.utils.data.DataLoader(
                        trainset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=2)

    testset = tv.datasets.CIFAR10(root=r'A:\DataSet', transform=transform)

    test_loader = t.utils.data.DataLoader(
                        testset,
                        batch_size=4,
                        shuffle=False,
                        num_workers=2)

    return train_loader, test_loader

if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = t.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader = data_load()

    for epoch in range(4):
        train_loss = 0
        for n, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (n + 1) % 2000 == 0:
                print("epoch-{}, batch-{:>5} loss: {:.3f}".format(epoch, n + 1, train_loss / 2000))
                train_loss = 0

        correct, total = 0, 0
        with t.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predict = t.max(outputs, 1)
                correct += (predict == labels).sum()
                total += labels.size(0)
        print(total)
        print(correct)
        print(f'epoch-{epoch} test accuracy: {correct.item() / total}')