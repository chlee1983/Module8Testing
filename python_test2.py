import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets

# importing dataset
ori_data_folder = './label'
train_data_folder = './train'
val_data_folder = './val'

train_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(root=train_data_folder, transform=train_transforms)
original_dataset = torchvision.datasets.ImageFolder(root=ori_data_folder, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=val_data_folder, transform=train_transforms)

'''To show the images loaded from the ImageFolder above
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    # the grid nrow determines the images per row
    grid = torchvision.utils.make_grid(images, nrow=4)
    # figure size is the size of the display image
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    print('labels: ', labels)


show_transformed_images(train_dataset)
'''

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
original_loader = torch.utils.data.DataLoader(original_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        print("Epoch number %d" %(epoch+1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * running_correct / total

        print("- Testing dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              %(running_correct, total, epoch_acc, epoch_loss))

        evaluate_model_on_test_set(model, test_loader)

    print("Finished")
    return model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    print("- Training dataset. Got %d out of %d images correctly (%.3f%%)"
              %(predicted_correctly_on_epoch, total, epoch_acc))


resnet18_model = models.resnet18(pretrained=True)
num_features = resnet18_model.fc.in_features
number_of_classes = 40
resnet18_model.fc = nn.Linear(num_features, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
# momentum = accelerate the gradient vectors in the right direction, leading to faster converging
# weight decay = help with prevent overfitting
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_nn(resnet18_model, train_loader, val_loader, loss_fn, optimizer, 10)
