import os.path
from src.features.train_test import train, test
import numpy as np
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

LOAD_TRUNCATED_IMAGES = True

data_dir = 'data/dog_images/'

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2
# n_epochs
n_epochs = 100

train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Apply transforms
train_data = datasets.ImageFolder(data_dir + 'train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transforms)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))

# remove image that is corrupted
indices.remove(5255)
num_train = num_train-1

np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader_transf = torch.utils.data.DataLoader(train_data,sampler=train_sampler,batch_size=batch_size, num_workers=num_workers)
validloader_transf = torch.utils.data.DataLoader(train_data, sampler=valid_sampler,batch_size=batch_size, num_workers=num_workers)
testloader_transf = torch.utils.data.DataLoader(test_data,batch_size=batch_size,num_workers=num_workers)

# check if class_names file exist
if not os.path.exists('class_names.csv'):
    class_names = [item[4:].replace("_", " ") for item in trainloader_transf.dataset.classes]
    class_names.to_csv('class_names.csv')

# check if cuda is available
use_cuda = torch.cuda.is_available()
 ('use_cuda',use_cuda)

# Load the pretrained model from pytorch
model_transf = models.resnet50(pretrained=True)

# unfreeze training for all "features" layers
for param in model_transf.parameters():
    param.requires_grad = False

# get number of inputs of previous layer
n_inputs = model_transf.fc.in_features

# update new classification layer
model_transf.fc = nn.Linear(n_inputs, 133)

if use_cuda:
    model_transf = model_transf.cuda()

torch.save(model_transf.state_dict(), 'model_transf.pt')

criterion_transf = nn.CrossEntropyLoss()
optimizer_transf = optim.Adam(model_transf.fc.parameters(), lr=0.0001)

# train model
model_transf = train(n_epochs, trainloader_transf, validloader_transf, model_transf, optimizer_transf, criterion_transf, use_cuda,
                     'src/models/model_transf.pt')

# test model
test(testloader_transf, model_transf, criterion_transf, use_cuda)
