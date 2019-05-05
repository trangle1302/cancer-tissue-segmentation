# Data augmentation by albumentations package
# For detail of the available augmenters, check out https://github.com/albu/albumentations

def transformations():
      data_transforms = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.RandomRotate90(p=0.5),
          albumentations.Transpose(p=0.5),
          albumentations.Flip(p=0.5),
          albumentations.OneOf([
              albumentations.CLAHE(clip_limit=2), albumentations.IAASharpen(), albumentations.IAAEmboss(), 
              albumentations.RandomBrightness(), albumentations.RandomContrast(),
              albumentations.JpegCompression(), albumentations.Blur(), albumentations.GaussNoise()], p=0.5), 
          albumentations.HueSaturationValue(p=0.5), 
          albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])

      data_transforms_test = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])

      data_transforms_tta0 = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.RandomRotate90(p=0.5),
          albumentations.Transpose(p=0.5),
          albumentations.Flip(p=0.5),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])

      data_transforms_tta1 = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.RandomRotate90(p=1),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])

      data_transforms_tta2 = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.Transpose(p=1),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])

      data_transforms_tta3 = albumentations.Compose([
          albumentations.Resize(224,224),
          albumentations.Flip(p=1),
          albumentations.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
          AT.ToTensor()
          ])
          
          return data_transforms, data_transforms_test, data_transforms_tta0, data_transforms_tta1, data_transforms_tta2, data_transforms_tta3
