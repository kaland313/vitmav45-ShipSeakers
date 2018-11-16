# Ship detection on satellite images using deep learning
The scripts found in this repository were written as the project assignment for the course Deep Learning in Practice with Python and LUA (VITMAV45). As part of this project, we aim to provide a solution for the Airbus Ship Detection Challenge organized by Kaggle. More information about the competition can be found on its website: https://www.kaggle.com/c/airbus-ship-detection

Team name: ShipSeakers

Team members:
* Csatlós, Tamás Péter (cstompet@gmail.com)
* Kalapos, András (kalapos.andras@gmail.com)


## Motivations
Earth observation using satellite data is a rapidly growing field. We use satellites to monitor polar ice caps, to detect environmental disasters such as tsunamis, to predict the weather, to monitor the growth of crops and many more. 
*Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.* (Quoted from the description of the kaggle challenge)

## Goals, description of the competition
We are given satellite images (more accurately sections of satellite images), which might contain ships or other waterborne vehicles. The goal is to segment the images to the "ship"/"no-ship" classes (label each pixel using these classes). The images might contain multiple ships, they can be placed close to each other (yet should be detected as separate ships), they can be located in ports, can be moving or stationary, etc. The pictures might show inland areas,the sea without ships, can be cloudy or foggy, lighting conditions can vary. 
The training data is given as images and masks for the ships (in a run length encoded format). If an image contains multiple ships, each ship has a separate record, mask. 

## Prerequisites
The training data can be downloaded from the competition's website after agreeing to the terms. Please note that the data might not be available after the submission deadline. 
We used Python 3.6 with Keras and Tensorflow and some other necessary packages. 

## Directory structure and files
The folder train_img contains all the images downloaded as the train_v2.zip
```
├── ShipDetectionDataPrep.ipynb
└── data/
    ├── train_ship_segmentations_v2.csv
    └── train_img/
└── Train Test Scripts/
    ├── Train.py
    ├── ShipSegFunctions.py
    ├── Test.py
    ├── model.hdf5
    ├── model.png
    └── test_img_ids.npy
```

## Running the data preparation script
The script can be executed if a few images are included in the train_img folder. Whithout this it can't show examples for different scenarios apperaring in the dataset. 

## The train and test scripts
The ShipSegFunctions.py script mostly conains functions and the generator class which are are to run the train and test scripts.

The Train.py sript loads and preprocesses the data for training, it includes network definitions and the training. The dataset is split into training and test partitions, but the testing is done in a separate file so the IDs of the test dataset are stored in test_img_ids.npy. The model is saved to model.hdf5 after every epoch which improves on the network performance.

Test.py atthe moment only produces predictions for a few images, later we intend to implement proper analysis of the test results. 

## The trained network
The trained network is a typical Unet architecture network, presented in the paper by Ronneberger (https://arxiv.org/abs/1505.04597), which is widely used for semantic segmentation. Due to memory limitations of the available GPU used during the first trainings the images were downsampled to 192x192 pixes and less layers were used compared to the original paper.
The structure of network is presented on the Train Test Scripts/model.png. 

## Results 
A network was trained for 10 epochs, during each epoch only 1600 images were shown to the network in bathces of 8. We used the adam optimizer with binary crossentropy lossfunction. 
The network managed to learn the basic ship recognition tast, but clearly needs significant improvements. One prediction it made can be seen on the ExamplePrediction.png. 
![Example predictions](https://github.com/kaland313/vitmav45-ShipSeakers/blob/master/ExamplePrediction.png)
