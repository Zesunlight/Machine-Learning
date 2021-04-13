import glob
import time
from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision import models
import csv
import numpy as np
import torch as t
from torch.utils import data
from torch.utils.data import DataLoader


class CatDogDataset(data.Dataset):
    def __init__(self, root_path):
        self.image_path = glob.glob(str(root_path / '*.jpg'))
        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])

    def __getitem__(self, item):
        path = self.image_path[item]
        label = 0 if 'cat' in path.split('\\')[-1] else 1
        pil_image = Image.open(path)
        torch_image = self.transform(pil_image)

        return torch_image, label

    def __len__(self):
        return len(self.image_path)


def load_data(data_set_path, batch_size=32, shuffle=True):
    dataset = CatDogDataset(data_set_path)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=6,
                        drop_last=False
                        )

    return loader


def train(epochs, batch_size, learning_rate):
    train_loader = load_data(Path(r'A:\DataSet\cat_dog\train'), batch_size)

    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(t.load(r'A:\DataSet\Parameters\catdog_transfer_para.pkl'))
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(resnet.parameters(), lr=learning_rate)
    exp_lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for epoch in range(epochs):
        resnet.train()
        train_loss, accuracy = 0.0, 0.0

        for index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = resnet(inputs)
            _, predicts = t.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            accuracy += (predicts == labels).sum().item()

            if (index + 1) % 100 == 0:
                print(f"index: {index + 1}, "
                      f"train_loss: {train_loss / (batch_size * (index + 1))}, "
                      f"accuracy: {accuracy / (batch_size * (index + 1))}, "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

        exp_lr_scheduler.step()
        train_loss = train_loss / (len(train_loader) * batch_size)
        accuracy = accuracy / (len(train_loader) * batch_size)
        print(f'epoch: {epoch}, train_loss: {train_loss}, accuracy: {accuracy}')
        t.save(resnet.state_dict(), r'A:\DataSet\Parameters\catdog_transfer_para_'+str(epoch)+'.pkl')


def test(batch_size=32):
    test_path = Path(r'A:\DataSet\cat_dog\test')
    test_loader = load_data(test_path, batch_size=batch_size, shuffle=False)

    net = models.resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 2)

    net = net.eval()
    net.load_state_dict(t.load(r'A:\DataSet\Parameters\catdog_transfer_para_3.pkl'))
    with open(r'A:\DataSet\cat_dog\result.csv', 'w+') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        for n, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predict = t.max(outputs, 1)
            predict = np.asarray(predict)
            index = [i for i in range(n * batch_size, (n + 1) * batch_size)]
            f_csv.writerows(list(zip(index, predict)))
            print(n)


if __name__ == '__main__':
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    test(64)
    # train(epochs=10, batch_size=96, learning_rate=0.0003)
