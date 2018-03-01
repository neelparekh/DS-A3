############################ Imports, housekeeping, and globals ############################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from torchvision.models.resnet import model_urls

import os
import pandas as pd
import numpy as np
import time
import random
import copy
import matplotlib.pyplot as plt



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
label_conversion = {'n09193705':0 , 'n09246464':1 , 'n09256479':2 , 'n09332890':3 , 'n09428293':4 }

############################ Set up how our model is trained ############################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


############################ Create datasets and dataloaders for train/val sets ############################
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = os.path.join(os.getcwd(),'tiny-imagenet-5')
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders    = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes  = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names    = image_datasets['train'].classes
print('number images: ',dataset_sizes)

use_gpu = torch.cuda.is_available()


########################### set up our pre-trained model and training params ############################
#################################### Use as fixed feature extractor #####################################

# print('Trying Frozen Weights (FFE):')

# model_conv = models.resnet50(pretrained=True)
# for param in model_conv.parameters():
# 	param.requires_grad = False
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, 5)

# if use_gpu:
# 	model_conv = model_conv.cuda()

# # Set loss function criterion
# criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as opposed to before.
# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.01, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=3, gamma=0.1)


# ########################### train our model ############################
# model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=50)
# torch.save(model_conv,'FFE1')

########################### set up our pre-trained model and training params ############################
########################################## Use as fine-tuning ###########################################

print('Trying Finetuning (FT):')

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.03, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

########################### train our model ############################
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(model_ft,'FT1')

######################################### Helper classes/methods #########################################

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'label': torch.Tensor(label)}

# class Norms(object):
#     """Normalize Numpy Arrays"""

#     def __call__(self, sample):
#         image, label = sample['image'].float(), sample['label']

#         mean = torch.Tensor([0.485, 0.456, 0.406])
#         std  = torch.Tensor([0.229, 0.224, 0.225])

#         temp = image - mean
#         image_norm = np.divide(temp, std)
#         return {'image': image_norm,
#                 'label': label}


# class RandomHorizontalFlip(object):
#     """Horizontally flip the given NumPy Image randomly with a probability of 0.5."""

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         """
#         Args:
#             img (NumPy Image): Image to be flipped.
#         Returns:
#             NumPy Image: Randomly flipped image.
#         """
#         if random.random() < 0.5:
#             image = np.flip(image, 1).copy()
#         return {'image': image,
#                 'label': label}

# class RandomVerticalFlip(object):
#     """Horizontally flip the given NumPy Image randomly with a probability of 0.5."""

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         """
#         Args:
#             img (NumPy Image): Image to be flipped.
#         Returns:
#             NumPy Image: Randomly flipped image.
#         """
#         if random.random() < 0.5:
#             image = np.flip(image, 0).copy()
#         return {'image': image,
#                 'label': label}


