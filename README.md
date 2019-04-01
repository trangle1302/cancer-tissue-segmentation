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

Phase 2: Data preparation and processing (DONE, as we decided to use camelyon16 on Kaggle): 
For camelyon16: already in 220,025 patches of 32x32px, only binary label, no mask https://www.kaggle.com/c/histopathologic-cancer-detection/data. 
For camelyon17: Download, find suitable storage (HPC/allowed access or AWS student instant, free?) that's assessible remotely
https://camelyon17.grand-challenge.org/Data/


Phase 3: Model implementation (2 weeks, meeting: 15/3/2019):  
(Continute) Literature study of available methods -> written report (all 3).  
Publish augmentation and model (nasnet, try other architectures as well) on github (Trang).  
Code review (Reda&Ethan, delayed to next stage after the code pipeline is more complete).  
Segmentation with lesion annotation: one of the winner's solutions (1 in 3 publications provided).  
Weakly labeled semantic segmentation (binary label): Attention net, Unet. 

Meeting 01/4/2019
Trang: implement more models, try MCOF if possible, put comments into code for review later. Update: Tried densenet169 and inceptionv3 in pytorch. 
Ethan: Combining the 3 literature review into final report format, download images from kaggle and upload into google drive for Colab.  Update: combined 3 reviews, but need to check the flow and consistency again. Downloaded kaggle images and uploaded on Drive. 
Reda: investigate the possibility of using Nextflow to elevate the multiple model ensemble later. Update: Nextflow is not usable for this project.


Meeting 12/4/2019
Trang: implement Resnet?, put comments. Start code review. 
Ethan: Adjust the report. Start code review. 
Reda: Check out unit tests (Dawid's links) and decompressing images from Colab. 


Phase 4: Data analysis, report writing (3 weeks, meeting: )
Report:
EDA: data description, class imbalance. 
Performance metrics: FROC, AUC, comparing this to results from state-of-the-art model at the moment. 
Unit tests: what are the tests that we conducted? Are they enough? What are other essential tests that was not possible for us to perform? Reason.
Generate nice images of segmented mask for reports. 
Discuss limits/disadvantages of our models. 
