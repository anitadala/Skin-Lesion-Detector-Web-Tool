# HAM10000SkinLesionDetector
Multi Class classification using CNN models on HAM10000 Skin Lesion Dataset



This repository contains all the models I have experimented with and I have also created flask based UI for the classification prediction.

Details about the files:
mymodel  :  Python File in which classification is done based on CNN model [MobileNetV2 Pre-trained model].The weights are then saved to the 'model12345.h5' file for directly used for UI purpose.

app.py : Flask based UI file which helps in prediction of the image by running the 'model12345.h5' file in the backend for making prediction by getting image by the user and predict the output.

base.html & index.html : For basic Interface.

uploads:It contains the test image example


# Skin-Lesion-Detector-Tool
 
 Here Datasets must be downloaded from the Kaggle.
 Link to download datasets : https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 , https://www.kaggle.com/discdiver/mnist1000-with-one-image-folder (contains all images in 1 folder only)
  Then extract the datasets with .csv file into same folder Skin-Lesion-Detector-Web Tool
      That is : HAM10000_metadata.csv
                HAM10000_images_part_1
                HAM10000_images_part_2
                These above should be extracted in that above folder.

Required Libraries :
    Web framework : Flask 
    Tensorflow
    Matplotlib
    Keras
    Numpy
    Pandas
    Sklearn
 These above libraries are mandatory.
 
 Steps to follow :
 Step 1 : Run the ‘ham1000-MobileNetV2.ipynb’ file in either Jupyter/Visual Studio Code
 Step 2 : At the final step of Training the model , save that model in the same folder in  which  the ‘app.py’ file is present.
 Step 3 : Give the path of saved Model in app.py 
 (e.g:  Model= load_model('model12345.h5') )
 Step 4 : Now run ‘app.py’ file to get the UI of Model. Follow the localhost link to open the User Interface in Web Browser
