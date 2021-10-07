import os
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

classes = ('我', '与', '相', '祖', '田', '九', '十', '二', '团', '.')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_dataset_path = './train'
# val_dataset_path = './validation'
original_dataset_path = './label'
batchSize = 216
datasets_transforms = transforms.Compose([transforms.ToTensor()])

training_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=datasets_transforms)
# validation_dataset = torchvision.datasets.ImageFolder(root=val_dataset_path, transform=datasets_transforms)
original_dataset = torchvision.datasets.ImageFolder(root=original_dataset_path, transform=datasets_transforms)

training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batchSize, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=32, shuffle=True)
original_loader = torch.utils.data.DataLoader(dataset=original_dataset, batch_size=batchSize, shuffle=True)

#
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# dataiter = iter(training_loader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%2s' % classes[labels[j]] for j in range(216)))
# for images, _ in training_loader:
#     tr_image_count_in_a_batch = images.size(0)
#     print("training images " + str(tr_image_count_in_a_batch))
#
# for images, _ in validation_loader:
#     tr_image_count_in_a_batch = images.size(0)
#     print("validation images " + str(tr_image_count_in_a_batch))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')