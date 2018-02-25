############################ Imports, housekeeping, and globals ############################
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.models.resnet import model_urls

import os
import pandas as pd
import numpy as np
import time
import random
import copy
from PIL import Image


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
label_conversion = {'n09193705':0 , 'n09246464':1 , 'n09256479':2 , 'n09332890':3 , 'n09428293':4 }

############################ classes to help load our images into PyTorch ############################
class TinyNetTrainDataset(Dataset):
	"""TinyNet dataset."""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): Directory with the images.
			image_filename (string): Name of the image file to be used.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		train_dir = os.path.join(root_dir,'train')
		labels = list(os.walk(train_dir))[0][1]
		img_dirs = [os.path.join(train_dir, a, 'images') for a in labels]

		imagepaths = []
		for i,label in enumerate(labels):
			# get all files in this label's sudirectory and append to the data structure
			subfiles = list(os.walk(img_dirs[i]))[0][2]
			i_paths = [ os.path.join(img_dirs[i],file) for file in subfiles ]
			for imagepath in i_paths:
				imagepaths.append((label_conversion[label],imagepath))
		self.image_paths = imagepaths
		self.root_dir = train_dir #.../train
		self.transform = transform


	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		label = self.image_paths[idx][0] # get idx image name from dataset
		image_path = self.image_paths[idx][1] # get idx image from dataset
		image = Image.open(image_path)
		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'label': label}
		return sample

class TinyNetValDataset(Dataset):
	"""TinyNet dataset."""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			root_dir (string): Directory with the images.
			image_filename (string): Name of the image file to be used.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		val_dir = os.path.join(root_dir, 'val')
		train_dir = os.path.join(root_dir, 'train')
		val_img_dir = os.path.join(val_dir, 'images')
		annotations_path = os.path.join(val_dir, 'val_annotations.txt')
		ann = pd.read_table(annotations_path,header=None)
		labels = list(os.walk(train_dir))[0][1]
		imagepaths = []
		for label in labels:
			imagepaths.append( [(label_conversion[label], os.path.join(val_img_dir,img)) for img in list(ann.loc[ann[1] == label].loc[:,0]) ] )
		self.image_paths = imagepaths
		self.root_dir = val_dir #.../val
		self.transform = transform


	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		label = self.image_paths[idx][0] # get idx image label from dataset
		image_path = self.image_paths[idx][1] # get idx image from dataset
		image = Image.open(image_path)
		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'label': label}
		return sample


############################ Helper classes/methods ############################

#None needed

############################ Set up how our model is trained ############################
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	sigm = nn.Sigmoid()
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
			# for data in dataloaders[phase]:
			for i,data in enumerate(dataloaders[phase]):
				# get the inputs
				inputs = data['image']
				label = data['label']

				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					label = Variable(label.cuda())
				else:
					inputs, label = Variable(inputs), Variable(label)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(inputs)
				# pred_probs = sigm(outputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, label)

				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()

				# statistics
				if i%100 == 0:
					print(i*batch_size)
				#   temp = torch.sum(preds == label.data)
				#   print('Acc: {}/{} = {:.2f} Percent'.format(temp,21*batch_size,temp/21/batch_size*100))
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == label.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

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
rd = '/Users/neelparekh/Cornell/DSitW/assignment3/DS-A3/tiny-imagenet-5'
TN_train = TinyNetTrainDataset(root_dir=rd,
								transform = transforms.Compose([
												transforms.RandomSizedCrop(224),
												transforms.RandomHorizontalFlip(),
												transforms.ToTensor(),
												transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											])
# , RandomVerticalFlip(), RandomHorizontalFlip()])
								)
TN_val = TinyNetValDataset(root_dir=rd,
								transform = transforms.Compose([ 
												transforms.Scale(256), 
												transforms.CenterCrop(224),
												transforms.ToTensor(),
												transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											])
								)
image_datasets = {'train': TN_train, 'val':TN_val}
print('example: ', TN_train[0])

batch_size = 4
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=4)  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
use_gpu = torch.cuda.is_available()


########################### set up our pre-trained model and training params ############################
#################################### Use as fixed feature extractor #####################################

# model_urls['resnet18'] = model_urls['resnet152'].replace('https://', 'http://')

model_conv = models.resnet34(pretrained=True)
for param in model_conv.parameters():
	param.requires_grad = False
num_ftrs = model_conv.fc.in_features
num_ftrs_out = model_conv.fc.out_features
model_conv.fc = nn.Linear(num_ftrs, 5)

if use_gpu:
	model_conv = model_conv.cuda()

# Set loss function criterion
criterion = nn.CrossEntropyLoss()

# print('Traceback (most recent call last):\n  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 40, in _worker_loop\n    samples = collate_fn([dataset[i] for i in batch_indices])\n  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 40, in <listcomp>\n    samples = collate_fn([dataset[i] for i in batch_indices])\n  File "assign3.py", line 62, in __getitem__\n    sample = self.transform(sample)\n  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/torchvision/transforms.py", line 34, in __call__\n    img = t(img)\n  File "assign3.py", line 108, in __call__\n    image, label = sample[\'image\'], sample[\'label\']\nKeyError: \'label\'\n')

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


########################### train our model ############################
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=50)
torch.save(model_conv,'FFE1')


# ############################ create model output with correct metrics ############################
# model18 = torch.load('model18bcelogits')
# # batch_size = 1434/2

# sigm = nn.Sigmoid()
# dfs_labs,dfs_preds,dfs_outs = [0]*86,[0]*86,[0]*86

# for i,data in enumerate(dataloaders['val']):
#   if i%15 == 0: print(i*batch_size)
#   # get the inputs
#   inputs = data['image']
#   labs = data['labels']

#   # wrap them in Variable
#   inputs = Variable(inputs)

#   # forward
#   outputs = model18(inputs.float()).data
#   predictions = sigm(outputs).round()

#   # save
#   outputs_numpy, predictions_numpy,labs_np = outputs.numpy(), predictions.numpy(),labs.numpy()
#   dfs_outs[i], dfs_preds[i], dfs_labs[i] = pd.DataFrame(outputs_numpy), pd.DataFrame(predictions_numpy), pd.DataFrame(labs_np)

#   # predictionsdf.to_csv('NN_predictions_{:0>3}.csv'.format(i))
#   # dfs[i] = pd.DataFrame(labs_np)
#   # predictionsdf.index +=1

# dfouts = pd.concat(dfs_outs)
# dfpred = pd.concat(dfs_preds)
# dflabs = pd.concat(dfs_labs)
# dfouts.to_csv('outs.csv')
# dfpred.to_csv('pred.csv')
# dflabs.to_csv('labs.csv')


#   # predictionsdf.to_csv('NN_predictions_{:0>3}.csv'.format(i))

#   # CM = [0]*21
#   # for j in range(21):
#   #   instrument = instrument_names[j]
#   #   truth_instr = truth_numpy[:,j]
#   #   predictions_instr = predictions_numpy[:,j]
#   #   # print('{}\n{}'.format(truth_instr, predictions_instr))
#   #   CM[j] = (instrument, confusion_matrix(truth_instr,predictions_instr) )

#   # print('CMs:\n{}'.format(CM))
#   # # if i%100 == 0: 
#   #   print(i*batch_size)
#   #   temp = torch.sum(pred_probs.data.round().double() == labels.data)
#   #   print('Acc: {}/{} = {:.2f} Percent'.format(temp, batch_size*21, temp/21/batch_size*100))
#   # running_corrects += torch.sum(pred_probs.data.round().double() == labels.data)



