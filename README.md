<a href="https://propulsion.academy/"><img src="https://static.swissdevjobs.ch/pictures/propulsion-academy-logo.png" title="Propulsion Academy" alt="Learn how to become a Data Scientist with Propulsion Academy"></a>

# Tooth Cementum Annulation (TCA)

This repository contains all the work done by Thomas Oliver and Tomasz Siczek for their final project to TCA using Convoluntional Neural Networks, Grad-CAM and Image Segmentation.

This project was done in collaboration with the department of Biological Anthropology at the University of Freiburg.

## Authors

[Thomas Oliver](https://bit.ly/35Hm1gg), [Tomasz Siczek](https://www.linkedin.com/in/tomasz-siczek/)

## Supervisors

[Badru Stanicki](https://www.linkedin.com/in/badrustanicki/), [Albin Plathottathil](https://www.linkedin.com/in/albin-plathottathil/)

## University of Freiburg Department of Biological Anthropology

[Professor Ursula Wittwer-Backofen](https://www.researchgate.net/profile/Ursula_Wittwer-Backofen)

## Motivation

The goal of the project is to create a model using a Convolutional Neural Network (CNN) that will take in the image of a Tooth Cementum and then run a regression analysis to predict the age of the individual.

Tooth Cementum is accessed by cutting a horizontal slice of a tooth and then placing it under a microscope. During manual counting, several slices are taken and counted, and are verified by a second counter. An example of such a cementum picture can be seen below.

<img src="http://www.jfds.org/articles/2015/7/3/images/JForensicDentSci_2015_7_3_215_172441_f9.jpg" title="Tooth Cementum Example" width="50%" height = '50%'>

There are numerous challenges to consider for TCA, both manually and when applying a CNN model. The pictures vary significantly in size, zoom, orientation and quality, and are heavily imbalanced in favour of individuals between 40 and 70. This because these are pictures of teeth removed for orthodontic reasons, so younger individuals are far less present within the dataset.

For manual counting this is is difficult enough, but it is far more a concern for neural networks and require a lot of preprocessing before training.

## Data

The images and the metadata was provided by the department of Biological Anthropology at the University of Freiburg. It consists of 2634 images taken from 576 individuals of different ages and origins.

To use the data, there are links found in the respective repositories for the images to download from Google drive. The main folder can be found [here](https://drive.google.com/drive/folders/1TEllOTUFYdipxyP55dRqSG1wfelepxN0).

## Requirements

Most of the code can be run in Google Colab and will not require anything beyond what is in the notebook. The helper scripts will also work within the colab notebooks they are loaded into without a requirements file.

IMPORTANT: For best results in this project, clone this repo into a google drive folder otherwise there will be problems with running the notebooks. Also, make sure to chance the initial directory path (found after mounting google drive) to where the project has been saved. Easiest means to do so is to navigate to the files section on the left-hand side of Google colab and then copy the correct file path.

## Structure

### SRC
This folder contains the Python notebooks and python helper files to run the project. They are number in the order they should be run in.

### data

This folder contains all the images used in the course of this project, including manually created masks for image segmentation. The folder structure has been maintained, but will need to be populated by following the links provided in the locations.txt files

### Graphs

This is a folder containing graphical representations of outputs from the model. They also have graphical represntation of model architecture

### Logs

This folder contains the training logs of all model runs over the course of this project

### Models

Any saved models and model checkpoints can be found in this folder here.

## How to work with this repo

### Part 1 (Optional).

Run the first [notebook](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/01_Data_extraction_and_cleaning.ipynb) to process the metadata into a clean CSV file and to unpack the images from a ZIP file.

This step has already been done and will not be necessary unless changes are made to what information is needed in the metadata CSV.

### Part 2

This [notebook](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/02_multi_input_model.ipynb) Is a first run of a multi-input model on the current picture set and metadata. The results will not be that impressive unless the pictures have been changed.

### Part 3

Run the [notebook](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/03_Grad_Cam.ipynb) to run a Grad-Cam on the Images. This will output a Grad-Cam Image that will show what the Xception model is looking at when run as a classification problem between 7 age bins. Useful if using new images.

### Part 4

This [notebook](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/04_Semantic_segmentation_attempt.ipynb) Will run a pre-trained image segmentation model on the images in the Image folder and create cementum-only images that can be used for the Vertically-sliced model in part 5. If you want to re-learn the model, uncomment the section that trains the images.

### Part 5

The model in this [notebook](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/05_Vertical_slicing_model.ipynb) will first split the images into 4 slices and take in additional training data (Tooth Code, Sex, and Eruption), learning for 50 epochs and then produce a scatterplot comparing true age versus predicted age.



### Part 6

This [notebook] (https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/tooth-annulation/-/blob/master/SRC/06_Vertical_slices_cropped_model.ipynb) is similar to the notebook described in section Part 5, but it does not use any additional features besides images to predict human age. It's an Xception model. This notebook also includes some discussions with regard to some specificities of the model. 
