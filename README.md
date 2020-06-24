# EMOTION-RECOGNITION-USING-DEEP-LEARNING-BY-KERAS
THE NEW TECHNIQUE FOR RECOGNITION OF FACIAL  EMOTION USING SUPERVISED LEARNING
This project aims to classify the emotion on a person's face into one of seven categories, using deep convolutional neural networks. This repository is an implementation of this research paper. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

Dependencies
Python 3, OpenCV, Tensorflow
To install the required packages, run pip install -r requirements.txt.
Basic Usage
The repository is currently compatible with tensorflow-2.0 and makes use of the Keras API using the tensorflow.keras library.

First, clone the repository and enter the folder
git clone https://github.com/Anees-baig/Emotion-detection.git
cd Emotion-detection
Download the FER-2013 dataset from here and unzip it inside the src folder. This will create the folder data.

If you want to train this model, use:

cd src
python emotions.py --mode train
If you want to view the predictions without training again, you can download the pre-trained model from here and then run:
cd src
python emotions.py --mode display
The folder structure is of the form:
src:

data (folder)
emotions.py (file)
haarcascade_frontalface_default.xml (file)
model.h5 (file)
This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.

Accuracy plot

Data Preparation (optional)
The original FER2013 dataset in Kaggle is available as a single csv file. I had converted into a dataset of images in the PNG format for training/testing and provided this as the dataset in the previous section.

In case you are looking to experiment with new datasets, you may have to deal with data in the csv format. I have provided the code I wrote for data preprocessing in the dataset_prepare.py file which can be used for reference.

Algorithm
First, the haar cascade method is used to detect faces in each frame of the webcam feed.

The region of image containing the face is resized to 48x48 and is passed as input to the CNN.

The network outputs a list of softmax scores for the seven classes of emotions.

The emotion with maximum score is displayed on the screen.
