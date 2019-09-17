import glob
import os
from pathlib import Path
from PIL import Image
from matplotlib import pyplot
from matplotlib.image import imread
from torch import nn
from torchvision import transforms as T
import csv
import numpy as np
import torch as t
from torch.utils import data
from torch.utils.data import DataLoader


def show_examples():
    folder = Path(r'A:\DataSet\cat_dog\train')

    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder / ('dog_' + str(i) + '.jpg')
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()


def data_statistic(dataset):
    total = len(dataset)
    height_mean, height_min, height_max = 0, 1000, 0
    width_mean, width_min, width_max = 0, 1000, 0

    for image, label in dataset:
        height_mean += image.size()[1]
        height_min = min(height_min, image.size()[1])
        height_max = max(height_max, image.size()[1])

        width_mean += image.size()[2]
        width_min = min(width_min, image.size()[2])
        width_max = max(width_max, image.size()[2])

    height_mean /= total
    width_mean /= total

    print(f'height_mean, height_min, height_max ---- {height_mean, height_min, height_max}')
    print(f'width_mean, width_min, width_max ---- {width_mean, width_min, width_max}')


class CatDogDataset(data.Dataset):
    def __init__(self, root_path):
        self.image_path = glob.glob(str(root_path / '*.jpg')) + glob.glob(r'A:\DataSet\cat_dog\val\*.jpg')
        self.transform = T.Compose([
            T.Resize(454),
            T.CenterCrop(454),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])

    def __getitem__(self, item):
        path = self.image_path[item]
        label = 0 if 'cat' in path.split('\\')[-1] else 1
        pil_image = Image.open(path)
        # np_image = np.asarray(pil_image)
        # torch_image = t.from_numpy(np_image)
        torch_image = self.transform(pil_image)

        return torch_image, label

    def __len__(self):
        return len(self.image_path)


def load_data(data_set_path, batch_size=32, shuffle=True):
    # data_statistic(train_dataset)
    # height_mean, height_min, height_max - --- (360.3091, 32, 768)
    # width_mean, width_min, width_max - --- (404.16525, 42, 1050)

    dataset = CatDogDataset(data_set_path)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        drop_last=False
                        )

    # validation_dataset = CatDogDataset(validation_path)
    # validation_loader = DataLoader(validation_dataset,
    #                                batch_size=batch_size,
    #                                shuffle=False,
    #                                num_workers=4,
    #                                drop_last=False
    #                                )

    return loader


class Net(nn.Module):
    def __init__(self, init_weights=True):
        super(Net, self).__init__()

        # 454
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 7, 3),  # 150
            nn.MaxPool2d(2, 2),  # 75
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 5, 2),  # 36
            nn.MaxPool2d(2, 2),  # 18
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3),  # 16
            nn.MaxPool2d(2, 2),  # 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = t.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        para = r'A:\DataSet\Parameters\catdog_para.pkl'
        if os.path.exists(para):
            self.load_state_dict(t.load(para))
            print('load parameters')
        else:
            for m in self.modules():
                if isinstance(m, t.nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


def train(net, current_loss=100000, batch_size=32, epochs=2, learning_rate=0.0001):
    train_path = Path(r'A:\DataSet\cat_dog\train')
    train_loader = load_data(train_path, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)

    net = net.train()
    for epoch in range(epochs):
        train_loss = 0
        for n, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).type(t.LongTensor)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (n + 1) % 125 == 0:
                print(f'processed with {n + 1} batches, train loss {train_loss / ((n + 1) * batch_size)}')

        print(f'epoch: {epoch}, train_loss: {train_loss:.4f}')
        if current_loss > train_loss:
            t.save(net.state_dict(), r'A:\DataSet\Parameters\catdog_para.pkl')
            print(f'save parameters with train loss {train_loss}')
            current_loss = train_loss


def validation(net, batch_size=32):
    validation_path = Path(r'A:\DataSet\cat_dog\val')
    validation_loader = load_data(validation_path, batch_size=batch_size, shuffle=False)

    validation_accuracy = 0
    net = net.eval()
    for n, (inputs, labels) in enumerate(validation_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        _, predict = t.max(outputs, 1)
        validation_accuracy += (predict == labels).sum().item()

    validation_accuracy /= len(validation_loader)
    print(f'validation accuracy {validation_accuracy}')

    return validation_accuracy


def test(net, batch_size=32):
    test_path = Path(r'A:\DataSet\cat_dog\test')
    test_loader = load_data(test_path, batch_size=batch_size, shuffle=False)

    net = net.eval()
    with open(r'A:\DataSet\cat_dog\result.csv', 'w+') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        for n, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predict = t.max(outputs, 1)
            predict = np.asarray(predict)
            index = [i for i in range(n * batch_size, (n + 1) * batch_size)]
            f_csv.writerows(list(zip(index, predict)))


if __name__ == '__main__':
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    model = Net().to(device)
    train(model, batch_size=32, epochs=15)
    validation(model, batch_size=64)
    test(model, batch_size=64)
