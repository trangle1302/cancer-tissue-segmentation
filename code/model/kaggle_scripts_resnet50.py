! pip install albumentations

# import libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time 
import tqdm
import random
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations
from albumentations import torch as AT

import scipy.special
from pytorchcv.model_provider import get_model as ptcv_get_model

from cancer_dataset import CancerDataset 
from data_transform import transformations 

cudnn.benchmark = True

# Seeding for reproducibility 
SEED = 123
base_dir = '../input/'
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

# Loading label and split training into tr and val
labels = pd.read_csv(base_dir+'train_labels.csv')
tr, val = train_test_split(labels.label, stratify=labels.label, test_size=0.10, random_state=SEED)
img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}

# Data transformations:

data_transforms, data_transforms_test, data_transforms_tta0, data_transforms_tta1, data_transforms_tta2, data_transforms_tta3 = transformations()

# Input image path to the dataloader to prepare for training
dataset = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
val_set = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms_test, labels_dict=img_class_dict)
test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_test)
train_sampler = SubsetRandomSampler(list(tr.index)) 
valid_sampler = SubsetRandomSampler(list(val.index))
batch_size = 24
num_workers = 0
# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)


# Defining densenet169: pretrained Densenet169 on Imagenet, and customized head
import torchvision.models as models

class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.linear = nn.Linear(self.model.fc.out_features, 32)
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ReLU()
        self.out = nn.Linear(32, 1)
    
    def forward(self, x):
        out = self.model(x)
        conc = self.linear(out)
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)

        res = self.out(conc)

        return res


model_conv = Resnet50(pretrained=False).cuda()


#### TRAINING
# To user pretrained model, I first freeze the pretrained weights to train for 1 epoch to activate the head (this helps making full use of the pretrain model)
# After freeze training for 1 epochs (training only the head), I unfreeze all weights and train the whole network for another 2 epochs 
# Loss = sigmoid + binary cross entropy 
# Adam optimizer
# Learning rate decay to 1/5 every 5 epochs (so it doesn't reduce in our training session, since I trained for a total of 3 epochs only)
# Only save the model with the lowest loss

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, 5, gamma=0.2)
scheduler.step()

val_auc_max = 0
patience = 25
# current number of tests, where validation loss didn't increase
p = 0
# whether training should be stopped
stop = False

# number of epochs to train the model
n_epochs = 1
for epoch in range(1, n_epochs+1):
    
    if stop:
        print("Training stop.")
        break
        
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    train_auc = []
        
    for tr_batch_i, (data, target) in enumerate(train_loader):
        
        model_conv.train()
        #time_s = time.time()

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output[:,0], target.float())
        train_loss.append(loss.item())
        
        a = target.data.cpu().numpy()
        try:
            b = output[:,0].detach().cpu().numpy()
            train_auc.append(roc_auc_score(a, b))
        except:
            pass

        loss.backward()
        optimizer.step()
        
        #time_e = time.time()
        #delta_t = time_e - time_s
        #print("training.... (time cost:%.3f s)"% delta_t)
        
        if (tr_batch_i+1)%600 == 0:    
            model_conv.eval()
            val_loss = []
            val_auc = []
            for val_batch_i, (data, target) in enumerate(valid_loader):
                data, target = data.cuda(), target.cuda()
                output = model_conv(data)

                loss = criterion(output[:,0], target.float())

                val_loss.append(loss.item()) 
                a = target.data.cpu().numpy()
                try:
                    b = output[:,0].detach().cpu().numpy()
                    val_auc.append(roc_auc_score(a, b))
                except:
                    pass

            print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.'%(epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss)) 
                  + '  train auc: %.4f, valid auc: %.4f'%(np.mean(train_auc),np.mean(val_auc)))
            train_loss = []
            train_auc = []
            valid_auc = np.mean(val_auc)
            if valid_auc > val_auc_max:
                print('Validation auc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                val_auc_max,
                valid_auc))
                #torch.save(model_conv.state_dict(), 'model_val%d.pt'%(valid_auc*10000))
                torch.save(model_conv.state_dict(), 'model.pt')
                val_auc_max = valid_auc
                p = 0
            else:
                p += 1
                if p > patience:
                    print('Early stop training')
                    stop = True
                    break   
            scheduler.step()
  
#Unfreeze
for idx, child in enumerate(model_conv.children()):
    if idx<0:
        for param in child.parameters():
            param.requires_grad = True
 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, 5, gamma=0.2)
scheduler.step()

val_auc_max = 0
patience = 25
# current number of tests, where validation loss didn't increase
p = 0
# whether training should be stopped
stop = False

# number of epochs to train the model
n_epochs = 2
for epoch in range(1, n_epochs+1):
    
    if stop:
        print("Training stop.")
        break
        
    print(time.ctime(), 'Epoch:', epoch)

    train_loss = []
    train_auc = []
        
    for tr_batch_i, (data, target) in enumerate(train_loader):
        
        model_conv.train()
        #time_s = time.time()

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output[:,0], target.float())
        train_loss.append(loss.item())
        
        a = target.data.cpu().numpy()
        try:
            b = output[:,0].detach().cpu().numpy()
            train_auc.append(roc_auc_score(a, b))
        except:
            pass

        loss.backward()
        optimizer.step()
        
        #time_e = time.time()
        #delta_t = time_e - time_s
        #print("training.... (time cost:%.3f s)"% delta_t)
        
        if (tr_batch_i+1)%600 == 0:    
            model_conv.eval()
            val_loss = []
            val_auc = []
            for val_batch_i, (data, target) in enumerate(valid_loader):
                data, target = data.cuda(), target.cuda()
                output = model_conv(data)

                loss = criterion(output[:,0], target.float())

                val_loss.append(loss.item()) 
                a = target.data.cpu().numpy()
                try:
                    b = output[:,0].detach().cpu().numpy()
                    val_auc.append(roc_auc_score(a, b))
                except:
                    pass

            print('Epoch %d, batches:%d, train loss: %.4f, valid loss: %.4f.'%(epoch, tr_batch_i, np.mean(train_loss), np.mean(val_loss)) 
                  + '  train auc: %.4f, valid auc: %.4f'%(np.mean(train_auc),np.mean(val_auc)))
            train_loss = []
            train_auc = []
            valid_auc = np.mean(val_auc)
            if valid_auc > val_auc_max:
                print('Validation auc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                val_auc_max,
                valid_auc))
                #torch.save(model_conv.state_dict(), 'model_val%d.pt'%(valid_auc*10000))
                torch.save(model_conv.state_dict(), 'model.pt')
                val_auc_max = valid_auc
                p = 0
            else:
                p += 1
                if p > patience:
                    print('Early stop training')
                    stop = True
                    break   
            scheduler.step()
   
# release gpu memory cache
torch.cuda.empty_cache()

# Switch model to evaluation/prediction mode 
model_conv.eval()

# Load the best weights (lowest loss during training)
saved_dict = torch.load('model.pt')
model_conv.load_state_dict(saved_dict)


##### Prediction: Test time augmentation
NUM_TTA = 32 # using 32 different types of augmentation during test/prediction time
sigmoid = lambda x: scipy.special.expit(x)

for num_tta in range(NUM_TTA):
    if num_tta==0:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    elif num_tta==1:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta1)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    elif num_tta==2:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    elif num_tta==3:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta3)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    elif num_tta<8:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms_tta0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    else:
        test_set = CancerDataset(datafolder=base_dir+'test/', datatype='test', transform=data_transforms)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    preds = []
    for batch_i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        output = model_conv(data).detach()
        pr = output[:,0].cpu().numpy()
        for i in pr:
            preds.append(sigmoid(i)/NUM_TTA)
    if num_tta==0:
        test_preds = pd.DataFrame({'imgs': test_set.image_files_list, 'preds': preds})
        test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])
    else:
        test_preds['preds']+=np.array(preds)
    print(num_tta)
    
# Saving submission file
sub = pd.read_csv('../input/sample_submission.csv')
sub = pd.merge(sub, test_preds, left_on='id', right_on='imgs')
sub = sub[['id', 'preds']]
sub.columns = ['id', 'label']
sub.to_csv('sub_tta.csv', index=False)
