# -*- coding: UTF-8 -*-
"""
=================================================
@Project: how_to_do
@File   : comic
@IDE    : PyCharm
@Author : Zhao Yongze
@Date   : 20/6/12
@Des    : in vain
=================================================="""
import glob
import os
import csv
import time
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ComicDataset(Dataset):
    def __init__(self, data_path, label_path='', train=True, transform=None):
        super(ComicDataset, self).__init__()

        self.data_path = data_path
        self.label_path = label_path
        self.train = train
        self.label = {}
        self.pictures_path = glob.glob(os.path.join(self.data_path, '*.jpg'))
        if self.train:
            self.read_csv_label()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = tv.transforms.Compose([
                # tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        image = Image.open(self.pictures_path[index]).convert("RGB")
        image = self.transform(image)
        if self.train:
            level = self.label[os.path.basename(self.pictures_path[index])]
            label = 0 if level <= 3 else 1
            return image, torch.tensor(label)
        else:
            return image, os.path.basename(self.pictures_path[index])

    def __len__(self):
        return len(self.pictures_path)

    def read_csv_label(self):
        with open(self.label_path, 'r') as file:
            lines = csv.reader(file, delimiter=',')
            next(lines)  # ignore the first line
            for line in lines:
                self.label[line[0]] = eval(line[1])
        return None


class Model(nn.Module):
    def __init__(self, num_classes=8):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=16, stride=16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3 * 4 * 4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 3 * 4 * 4)
        x = self.classifier(x)
        return x


def train(model, learning_rate, train_loader, num_epochs, device):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    print('start train {}'.format(time.asctime(time.localtime(time.time()))))
    for epoch in range(num_epochs):
        if epoch >= 4:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate / 10)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {}'
                      .format(epoch + 1, num_epochs,
                              i + 1, total_step,
                              loss.item(),
                              time.asctime(time.localtime(time.time()))))


def validate(model, valid_loader, device):
    # Test the model
    print('start valid {}'.format(time.asctime(time.localtime(time.time()))))
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print('Test Accuracy: {} %'.format(acc))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model/comic_gender_resnet18_{}.ckpt'.format(int(acc)))


def test(model, test_loader, device):
    print('start test {}'.format(time.asctime(time.localtime(time.time()))))
    model.eval()
    with torch.no_grad():
        with open(r'A:\Contest\Accelerator\test_gender.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['image', 'gender_status'])
            for image, name in test_loader:
                image = image.to(device)
                output = model(image)
                predicted = torch.argmax(output.data)
                f_csv.writerow((name[0], predicted.item()))
    return None


def gender():
    comic_train_data_path = r''
    comic_test_data_path = r''
    comic_train_label_path = r''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 3
    num_classes = 2
    batch_size = 8
    learning_rate = 0.0001

    comic_train_data = ComicDataset(comic_train_data_path, comic_train_label_path)
    comic_test_data = ComicDataset(comic_test_data_path, train=False)
    train_data_loader = DataLoader(
        dataset=comic_train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_data_loader = DataLoader(dataset=comic_test_data, batch_size=1)

    model = tv.models.resnet18(pretrained=True)
    num_in_feature = model.fc.in_features
    model.fc = nn.Linear(num_in_feature, num_classes)
    # model.load_state_dict(torch.load('model/comic_resnet34_9_97.ckpt'))
    # model = Model(num_classes)
    # summary(model, (3, 256, 256))
    train(model, learning_rate, train_data_loader, num_epochs=num_epochs, device=device)
    validate(model, train_data_loader, device)
    test(model, test_data_loader, device)
    print('finished {}'.format(time.asctime(time.localtime(time.time()))))


def predict(level_path, gender_path, save_path):
    with open(level_path, 'r') as l:
        with open(gender_path, 'r') as g:
            level = csv.reader(l, delimiter=',')
            sex = csv.reader(g, delimiter=',')
            next(level)  # ignore the first line
            next(sex)
            with open(save_path, 'w', newline='') as s:
                f_csv = csv.writer(s)
                f_csv.writerow(['image', 'gender_status'])
                for a, b in zip(level, sex):
                    f_csv.writerow((a[0], int(a[1]) + int(b[1]) * 4))


if __name__ == '__main__':
    predict(r'A:\Contest\Accelerator\test_level.csv',
            r'A:\Contest\Accelerator\test_gender.csv',
            r'A:\Contest\Accelerator\test.csv')
