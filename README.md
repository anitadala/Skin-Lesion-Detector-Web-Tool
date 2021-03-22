# HAM10000SkinCancerClassification
Multi Class classification using CNN models on HAM10000 Skin cancer Dataset



This repository contains all the models I have experimented with and I have also created flask based UI for the classification prediction.

Details about the files:
mymodel  :  Python File in which classification is done based on CNN model [MobileNet Pre-trained model].The weights are then saved to the 'model.h5' file for directly used for UI purpose.

app.py : Flask based UI file which helps in prediction of the image by running the 'model.h5' file in the backend for making prediction by getting image by the user and predict the output.

base.html & index.html : For basic Interface.

uploads:It contains the test image example

