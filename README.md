# cancer-tissue-segmentation
Cancer Tissue Segmentation

The goal of this project is segmenting histopathology images to identify cancer lesions.


Data: We will be using parts of Camelyon dataset with xxx images of size xxx (decide between Camelyon16 or 17)

Model: We will implement two models: 
1. Segmentation based on lesion annotation:
2. Segmentation based binary label: U-Net


Tasks: 
Phase 1: Literature search (2 weeks, next meeting: 21/02/2019)
Literature study of available methods -> written report

Phase 2: Data preparation and processing (3 weeks, meeting: )
For camelyon16: already in 220,025 patches of 32x32px, only binary label, no mask https://www.kaggle.com/c/histopathologic-cancer-detection/data
For camelyon17: Download, find suitable storage (HPC/allowed access or AWS student instant, free?) that's assessible remotely
https://camelyon17.grand-challenge.org/Data/

Phase 3: Model implementation (4 weeks, meeting: )
Data augmentation (packages
Segmentation with lesion annotation: one of the winner's solutions (1 in 3 publications provided) 
Weakly labeled semantic segmentation (binary label): Attention net, Unet

Phase 4: Data analysis, report writing (3 weeks, meeting: )
Report:
Performance metrics: FROC, AUC, comparing this to results from state-of-the-art model at the moment
Generate nice images of segmented mask for reports
Discuss limits/disadvantages of our models
