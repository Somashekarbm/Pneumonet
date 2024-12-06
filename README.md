# Pneumonet

Pneumonia Detection through chest scan images : Image Classification w/ Convolutional Neural Networks and Transfer Learning using ResNet and fine tuned resnet models.

## Overview
This was a project we made as an assignment after our 5th semester Computer Vision workshop.

The goal of this project is to utilize Convolutional Neural Networks (CNNs) on Chest X-Ray images to identify samples from patients with Pneumonia. The dataset used (version 3) comprises two folders representing the train set and the test set. The train folder is later partitioned in the notebook into train/validation sets.

Three different approaches are explored for image classification:

A simple CNN
Transfer Learning, employing a pretrained model with frozen layers for feature extraction
Fine Tuning, involving unfreezing the last layers of the pretrained model

Please note that the dataset used is the third version of the Chest X-Ray dataset available here.-> "https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images/data"

Content-
Dataset Information
Importing Packages and Dataset
Exploring the Data
Preparing the Data
Custom Model
Transfer Learning
Fine Tuning
Performance Metrics
References

Dataset Information
The dataset contains 5,856 validated Chest X-Ray images, split into a training set and a testing set of independent patients. Images are labeled as (disease:NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients aged one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

Importing Packages and Dataset
The notebook begins with the necessary imports of libraries and loading of the dataset. Various libraries such as Pandas, Matplotlib, NumPy, Seaborn, and TensorFlow are imported for data handling, visualization, and modeling purposes.

Exploring the Data
Data exploration includes analyzing the distribution of classes in both the training and testing sets, as well as displaying sample images from each class to gain insights into the dataset.

Preparing the Data
Data preparation involves creating a validation set, defining data generators for image augmentation, and setting up the data for model training.

Custom CNN
A custom CNN model is built from scratch, and its performance is evaluated on the dataset.

Transfer Learning
A pretrained ResNet152V2 model is used as a feature extractor, and a custom head is added to perform binary classification on the dataset.

Fine Tuning
The last few layers of the pretrained ResNet152V2 model are unfrozen, and the model is fine-tuned on the dataset.

Performance Metrics
Performance metrics such as accuracy, confusion matrix, classification report, and ROC-AUC curve are computed and visualized to evaluate the models' performance.

References
Tutorial on Keras flow_from_dataframe
TensorFlow documentation on Convolutional Neural Networks
TensorFlow documentation on Transfer Learning
Keras documentation on Applications
Keras documentation on ResNet152V2

conclusion-
This README provides an overview of the project, details about the dataset, steps involved in data preprocessing, model implementation, and evaluation metrics, along with references for further exploration.
