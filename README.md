# cancer-tissue-segmentation
## Cancer Tissue Segmentation

Part of CONTRA: Agile Bioinformatics for Agile Science (Krakow, 2-8 February 2018) organized by Ardigen

The goal of this project is segmenting histopathology images to identify cancer lesions.


Data: We will be using parts of Camelyon dataset with xxx images of size xxx (decide between Camelyon16 or 17)

Model: We will implement two models: 
1. Segmentation based on lesion annotation:
2. Segmentation based binary label: U-Net


Tasks: 
Phase 1: Literature search (3 weeks, meeting: 28/02/2019): 
Literature study of available methods -> written report

# Phase 2: Data preparation and processing (DONE, as we decided to use camelyon16 on Kaggle): 
For camelyon16: already in 220,025 patches of 32x32px, only binary label, no mask https://www.kaggle.com/c/histopathologic-cancer-detection/data. 
For camelyon17: Download, find suitable storage (HPC/allowed access or AWS student instant, free?) that's assessible remotely
https://camelyon17.grand-challenge.org/Data/


# Phase 3: Model implementation (2 weeks, meeting: 15/3/2019):  
(Continute) Literature study of available methods -> written report (all 3).  
Publish augmentation and model (nasnet, try other architectures as well) on github (Trang).  
Code review (Reda&Ethan, delayed to next stage after the code pipeline is more complete).  
Segmentation with lesion annotation: one of the winner's solutions (1 in 3 publications provided).  
Weakly labeled semantic segmentation (binary label): Attention net, Unet. 

# Meeting 01/4/2019 
Trang: implement more models, try MCOF if possible, put comments into code for review later. Update: Tried densenet169 and inceptionv3 in pytorch. 
Ethan: Combining the 3 literature review into final report format, download images from kaggle and upload into google drive for Colab.  Update: combined 3 reviews, but need to check the flow and consistency again. Downloaded kaggle images and uploaded on Drive. 
Reda: investigate the possibility of using Nextflow to elevate the multiple model ensemble later. Update: Nextflow is not usable for this project.


# Meeting 17/4/2019 
Trang: implement Resnet?, put comments. Start code review. 
Ethan: Adjust the report. Start code review. 
Reda: Check out unit tests (Dawid's links) and decompressing images from Colab. 

# To-do list until meeting 2/5/2019:
1. Unit test implementation (Trang marks down the code segments to be tested and Reda write the testing with mltest)
  a. Output of network if the last layer is probability should be a float between 0-1, if the last layer is logistic then the output should be binary.
  b. Variables change (input-output should not be in the same, loss should reduce, acccuracy should increase)
  c. Variables don't change (learning rate if fixed learning rate or ReduceLROnPlateau or LRStep don't apply).
2. Code review: put code into modules (Ethan, Reda), Test if the modular pipeline work on Kaggle by imputing code as dataset (Trang)
3. Final report: Summarizing background, methods (in reports):
  a. Background: Question addressed and challenges (identifying tumor vs not, like intro or background of 3 papers we reviewed), Existing methods (methods of the 3 papers). (Ethan)
  b. Method: Brief intro of all methods we used (InceptionV3, Resnet, NasnetMobile) (Trang or sb else who volunteer!)
  c. Result: Accuracy of each  model, compare performance with other exisitng participants in the challenge 
  d. Reference:
  


# Phase 4: Data analysis, report writing (3 weeks, meeting: )
Report:
EDA: data description, class imbalance. 
Performance metrics: FROC, AUC, comparing this to results from state-of-the-art model at the moment. 
Unit tests: what are the tests that we conducted? Are they enough? What are other essential tests that was not possible for us to perform? Reason.
Generate nice images of segmented mask for reports. 
Discuss limits/disadvantages of our models. 



## Last to-do list!!!
# Github repo
1. Model code (done)
2. Code review (Reda & Ethan)
3. Backlog (done)
4. Unit test (Reda)

# Report 
Introduction: Background, problems, literature review of available methods (Ethan & Reda)
Method: summarise model arch (Trang)
Results: Accuracy of 4 models + ensembles (Trang)
Reference (cite at the end!)

# Presentation (ppt slides)
Introduction: Problem (1-2 slides), Existing method (1 slide) (Ethan & Reda)
Method: Summarise the 4 models that we used (Resnet50, InceptionV3, Densenet169, Nasnetmobile) (Trang)
Results: Performance of 4 individual models + ensemble (Trang)


