import mltest
import tensorflow as tf
import numpy as np
import random
from kaggle_scripts_resnet50 import Resnet50 
from cancer_dataset import CancerDataset

def setup():
  mltest.setup()

def resnet50_mltest_suite():
  
  dataset = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
  train_sampler = SubsetRandomSampler(list(tr.index)) 
  batch_size = 24
  num_workers = 0
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
  (data, target) = train_loader[1]
  #data, target = data.cuda(), target.cuda() #if testing in CPU, remove this line
  model_conv = Resnet50(pretrained=True).cuda()
  model_conv.train()
  output = model_conv(data)
  random.seed(123)
  feed_dict = {
      input_tensor: np.random.normal(size=(10, 100)),
      label_tensor: np.random.randint((100))
  }
  setup()
  mltest.test_suite(
      data,
      output,
      feed_dict=feed_dict)
