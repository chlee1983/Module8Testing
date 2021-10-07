from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets

# importing dataset
ori_data_folder = './label'
train_data_folder = './train'

