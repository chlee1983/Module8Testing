import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


train_dataset_path = "./Module8-Project/label"

# image = (image - mean) / std

mean = [0.8353, 0.8353, 0.8353]
std = [0.3646, 0.3646, 0.3646]

train_transforms = transforms.Compose([
    transforms.Resize((102, 102)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


def set_devices():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    return torch.device(dev)


def train_nn(model, train_loader, criterion, optimizer, n_epochs):
    pass


resnet18_model = models.resnet18(pretrained=False)
num_ftrs = resnet18_model.fc.in_features
numbers_of_classes = 650
resnet18_model.fc == nn.Linear(num_ftrs, numbers_of_classes)

# def show_transformed_images(dataset):
#     loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
#     batch = next(iter(loader))
#     images, labels = batch
#
#     grid = torchvision.utils.make_grid(images, nrow=3)
#     plt.figure(figsize=(11, 11))
#     plt.imshow(np.transpose(grid, (1, 2, 0)))
#     plt.show()
#     print('label: ', labels)
#
#
# show_transformed_images(train_dataset)


