import os
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_dataset_path = './train'
val_dataset_path = './validation'
original_dataset_path = './label'

datasets_transforms = transforms.Compose([transforms.ToTensor()])

training_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=datasets_transforms)
validation_dataset = torchvision.datasets.ImageFolder(root=val_dataset_path, transform=datasets_transforms)
original_dataset = torchvision.datasets.ImageFolder(root=original_dataset_path, transform=datasets_transforms)


# training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=32, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=32, shuffle=True)
# original_loader = torch.utils.data.DataLoader(dataset=original_dataset, batch_size=32, shuffle=True)


# for images, _ in training_loader:
#     tr_image_count_in_a_batch = images.size(0)
#     print("training images " + str(tr_image_count_in_a_batch))
#
# for images, _ in validation_loader:
#     tr_image_count_in_a_batch = images.size(0)
#     print("validation images " + str(tr_image_count_in_a_batch))


class MyNN(Dataset):
    def __init__(self, x, y):
        x = x.float()
        x = x.view(-1, 102 * 102)
        self.x, self.y = x, y

    """contains logic for what should be returned when ask for the ix-th data points
    (ix will be an integer between 0 and __len__)"""

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    # specify the number of data points in the __len__ method (length of x)
    def __len__(self):
        return len(self.x)


def get_data():
    train = MyNN(training_dataset, original_dataset)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl


def get_model():
    model = nn.Sequential(nn.Linear(28 * 28, 1000),
                          nn.ReLU(),
                          nn.Linear(1000, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer


def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    # call the model like any python function on your batch of inputs
    prediction = model(x)
    # compute loss
    batch_loss = loss_fn(prediction, y)
    # based on the forward pass in model(x) compute all the gradients of 'model.parameters()'
    batch_loss.backward()
    # apply new weights = f(old-weights, old-weight-gradients) where "f" is the optimizer
    optimizer.step()
    # flush gradients memory for next batch of calculations
    optimizer.zero_grad()
    return batch_loss.item()


trn_dl = get_data()
model, loss_fn, optimizer = get_model()
losses, accuracies = [], []

#
# @torch.no_grad()
# def accuracy(x, y, model):
#     model.eval()
#     # get the prediction matrix for a tensor of 'x' images
#     prediction = model(x)
#     # compute if the location of maximum in each row coincides with ground truth
#     max_values, argmaxes = prediction.max(-1)
#     is_correct = argmaxes == y
#     return is_correct.cpu().numpy().tolist()
#
#
# for epoch in range(5):
#     print(epoch)
#     epoch_losses, epoch_accuracies = [], []
#     for ix, batch in enumerate(iter(trn_dl)):
#         x, y = batch
#         batch_loss = train_batch(x, y, model, optimizer, loss_fn)
#         epoch_losses.append(batch_loss)
#     epoch_loss = np.array(epoch_losses).mean()
#     for ix, batch in enumerate(iter(trn_dl)):
#         x, y = batch
#         is_correct = accuracy(x, y, model)
#         epoch_accuracies.extend(is_correct)
#     epoch_accuracy = np.mean(epoch_accuracies)
#     losses.append(epoch_loss)
#     accuracies.append(epoch_accuracy)
#
# epochs = np.arange(5) + 1
# plt.figure(figsize=(20, 5))
# plt.subplot(121)
# plt.title('Loss value over increasing epochs')
# plt.plot(epochs, losses, label='Training Loss')
# plt.legend()
# plt.subplot(122)
# plt.title('Accuracy value over increasing epochs')
# plt.plot(epochs, accuracies, label='Training Accuracies')
# plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
# plt.legend()
# plt.show()
