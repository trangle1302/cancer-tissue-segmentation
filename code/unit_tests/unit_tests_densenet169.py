import mltest
import kaggle_scripts_densenet169
import tensorflow as tf
import numpy as np

def setup():
  mltest.setup()

dataset = CancerDataset(datafolder=base_dir+'train/', datatype='train', transform=data_transforms, labels_dict=img_class_dict)
train_sampler = SubsetRandomSampler(list(tr.index)) 
batch_size = 24
num_workers = 0
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)


from my_model import Densenet169

def densenet169_mltest_suite():
  
  (data, target) = train_loader[1]
  #data, target = data.cuda(), target.cuda() #if testing in CPU, remove this line
  model_conv = DenseNet169(pretrained=True).cuda()
  model_conv.train()
  output = model_conv(data)
  mltest.test_suite(
      data,
      output,
      feed_dict=feed_dict)

def test_range():
  model = kaggle_scripts_densenet169.build_model()
  mltest.test_suite(
    model.logits, #output (= model_conv(data)) should be between 0,1
    model.train_op,
    output_range=(0,1))
