import torch as t
import torch.nn as nn
import torchvision as tv


def load_data(batch_size=16):
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    train_set = tv.datasets.MNIST(root=r'A:\DataSet\Mnist', transform=transform, train=True)
    print(train_set)
    train_loader = t.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_set = tv.datasets.MNIST(root=r'A:\DataSet\Mnist', transform=transform, train=False)
    print(test_set)
    test_loader = t.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader


class Flatten(nn.Module):
    # def __init__(self):
    #     super(Flatten, self).__init__()

    def forward(self, into):
        return into.view(into.size(0), -1)


def run_model(batch_size=32, learning_rate=0.001, epochs=3):
    train_loader, test_loader = load_data(batch_size)

    net = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 5, 3),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(5 * 11 * 11, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
    )

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        net = net.train()
        train_loss, train_accuracy = 0, 0
        for n, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).type(t.LongTensor)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predict = t.max(outputs, 1)
            train_accuracy += (predict == labels).sum().item() / batch_size

        train_loss /= n + 1
        train_accuracy /= n + 1
        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}')

        net = net.eval()
        test_accuracy = 0
        for n, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predict = t.max(outputs, 1)
            test_accuracy += (predict == labels).sum().item() / batch_size

        test_accuracy /= n + 1
        print(f'epoch: {epoch}, test_accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    BATCH_SIZE = 16
    run_model()
