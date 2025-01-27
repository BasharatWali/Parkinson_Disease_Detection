# Parkinson_Disease_Detection

This repository demonstrates a machine learning pipeline for detecting Parkinson’s Disease from 3D MRI scans. The project integrates data from the PPMI (Parkinson’s Progression Markers Initiative) dataset and the IXI dataset, performing the following key steps:      
  1. Data Conversion: Converts DICOM files to NIfTI format using dcm2niix.    
  2. Registration: Aligns and standardizes images to a common space.     
  3. Feature Extraction: Employs PyRadiomics to extract a rich set of radiomic features.
 
All the pre-processing was done on 3D nifti images.

Modeling: Implements a hybrid approach with         
  1. Random Forest + SVM (via a Scikit-learn Pipeline)         
  2. Gaussian Naive Bayes (GaussianNB)

By combining extensive preprocessing, feature engineering, and robust machine learning models, the repository aims to accurately distinguish Parkinson’s Disease subjects from healthy controls, contributing to improved diagnostics and research in neurodegenerative conditions.
