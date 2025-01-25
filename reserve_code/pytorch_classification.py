import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data import DataLoader, Dataset
#from ignite.engine import Engine
#from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from PIL import Image


# CV2 — это библиотека для работы с изображениями в Python. Она предоставляет множество функций для выполнения различных
# операций с изображениями, включая изменение размера, обрезку, поворот и другие преобразования

# labels = ['buildings', 'forest', 'glacier', 'mountain','sea', 'street']
img_size = 150
batch_size = 32


class IntelImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


image_dir = 'archive/seg_train/seg_train'
test_image_dir = 'archive/seg_test/seg_test'
pred_image_dir = 'archive/seg_pred/seg_pred'

categories = os.listdir(image_dir)

image_paths = []
labels = []
test_image_paths = []
test_labels = []


def set_dataset(list_images, list_labels, image_dir):
  for label, category in enumerate(categories):
      category_dir = os.path.join(image_dir, category) # к пути добавляется название папки
      for filename in os.listdir(category_dir): # для всех файлов в папке
          if filename.endswith('.jpg'):         # если это картинка(.jpg)
              list_images.append(os.path.join(category_dir, filename)) # в массив с путями добавить
              list_labels.append(label)              # в массив с метками добавить


set_dataset(image_paths, labels, image_dir)  # обучающий датасет
set_dataset(test_image_paths, test_labels, test_image_dir) # тестовый датасет

label_encoder = LabelEncoder() # Закодируйте целевые метки со значением от 0 до n_классов-1(в ).
                    # Этот преобразователь следует использовать для кодирования целевых значений, т. е. y, а не входных X.
labels = label_encoder.fit_transform(labels) #.fit_transform() Установите кодировщик меток и верните закодированные меток.
test_labels = label_encoder.transform(test_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

print('val_lab - ', val_labels)
print('val_lab_shape - ', val_labels.shape)

transform_train = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(150),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


train_dataset = IntelImageDataset(train_paths, train_labels, transform=transform_train)
val_dataset = IntelImageDataset(val_paths, val_labels, transform=transform_train)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print('val_loader: ', val_loader.dataset)
test_dataset = IntelImageDataset(test_image_paths, test_labels, transform=transform_train)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#dataiter = iter(train_loader)  # Функция iter() в Python используется для получения итератора у итерируемых объектов, например списков, кортежей, строк
#images, labels = next(dataiter)  # next() в Python возвращает следующий элемент из итератора. Если итератор исчерпан, он возвращает значение по умолчанию,

#imshow(torchvision.utils.make_grid(images))

print(' '.join(f'{categories[labels[j]]:5s}' for j in range(batch_size)))
##  МОДЕЛЬ

class Net(nn.Module):
    def __init__(self):
        super().__init__()
                                      # Conv2d Применяет двумерную свёртку к входному сигналу, состоящему из нескольких входных плоскостей.
        self.conv1 = nn.Conv2d(3, 32, 5) # параметры: 3 - Количество каналов во входном изображении, 6 - количество каналов, создаваемых свёрткой, 5 - размер ядра свёртки
        self.pool = nn.MaxPool2d(2, 2) #  MaxPool2 Применяет 2D-объединение по максимуму к входному сигналу, состоящему из нескольких входных плоскостей.
        # Параметры: kernel_size (Union[int, Tuple[int, int]]) — размер окна для вычисления максимума, stride (Union[int, Tuple[int, int]]) –  шаг окна. Значение по умолчанию — kernel_size
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 289 * 4, 1200) # Применяет аффинное линейное преобразование к входящим данным: y=xA^T+b.
        # Параметры in_features (int) – размер каждой входной выборки, out_features (int) – размер каждой выходной выборки,смещение (bool) — если установлено значение False, слой не будет
        # обучаться с учётом смещения. По умолчанию: True
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = x.view(-1, 64 * 289 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# оптимизатор
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # lr=0.001
# обучение
for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# Давайте быстро сохраним нашу обученную модель:

PATH = './intel_image_v_0_1.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(val_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{categories[labels[j]]:5s}' for j in range(batch_size)))

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print(predicted)
print(type(predicted))
print('Predicted: ', ' '.join(f'{categories[predicted[j]]:5s}'
                              for j in range(batch_size)))
correct = 0
total = 0
false_positives = 0
false_negatives = 0
# поскольку мы не тренируемся, нам не нужно вычислять градиенты для наших выходных данных
with torch.no_grad(): # torch.no_grad() — это контекстный менеджер в PyTorch, который отключает вычисление градиентов.
                      # Он полезен на этапах оценки или тестирования модели, когда не нужно вычислять градиенты, что
                      # экономит память и вычислительные ресурсы.
    acc_ = Accuracy(task='multiclass', num_classes=6)
    precis_ = Precision(task='multiclass', num_classes=6)
    recall_ = Recall(task='multiclass', num_classes=6)
    f1_ = F1Score(task='multiclass', num_classes=6)

    for data in val_loader:
        images, labels = data
        # вычисляйте выходные данные, прогоняя изображения по сети
        outputs = net(images)
        # класс с самой высокой энергией - это то, что мы выбираем в качестве прогноза
        _, predicted = torch.max(outputs, 1)

        acc_(predicted, labels)
        precis_(predicted, labels)
        recall_(predicted, labels)
        f1_(predicted, labels)

        accuraty_value = acc_.compute()
        precision_value = precis_.compute()
        recall_value = recall_.compute()
        f1_value = f1_.compute()
       # print(f'Accuracy - {100*float(accuraty_value):.1f}')
       # print(f'Precision - {100 * float(precision_value):.1f}')
       # print(f'Recall - {100 * float(recall_value):.1f}')
       # print(f'f1 - {100 * float(f1_value):.1f}')

        correct += (predicted == labels).sum().item()
        false_positives += torch.logical_and(predicted == 1, labels == 0).sum().item()
        false_negatives += torch.logical_and(predicted == 0, labels == 1).sum().item()

        total += labels.size(0)

    # print(pred_arr)
accuracy = 100 * correct // total
precision = 100 * correct // (correct + false_positives)
recall = 100 * correct // (correct + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
print(f'Precision of the network on the 10000 test images: {precision} %')
print(f'Recall of the network on the 10000 test images: {recall} %')
print(f'Recall of the network on the 10000 test images: {f1_score}')

# приготовьтесь подсчитывать прогнозы для каждого класса
correct_pred = {classname: 0 for classname in categories}
total_pred = {classname: 0 for classname in categories}

# again no gradients needed
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        #print('images - ', images)
        outputs = net(images)

        #print('outputs - ', outputs)
        #accuracy = Accuracy(task='multiclass', num_classes=6)
        #print('ACCURACY - ', accuracy(outputs, images))
        _, predictions = torch.max(outputs, 1)

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[categories[label]] += 1
            total_pred[categories[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')