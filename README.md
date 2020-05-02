# Image Captioning using COCO and Flickr Data

This project was implemented by:

* Ashwini Marathe
* Akshay Punwatkar
* Anshupriya Srivastava
* Srishti Saha

and was submitted as our final project for the course ECE 590 (Data Analysis at Scale in the Cloud).

## Objective of the Application

The applications takes an image file as an input and uses a Machine Learning model to generate a caption for the image. 

### Data Source

The training and test data for the application and the demo are as follows:
* COCO dataset: [link](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)


### Model

The model was trained using Tensorflow and the methodology was based on the notebook provided by Google Colab. The basic steps in training this model were as follows:

* Tokenize vocabulary from Training captions data
* Implement a seq2seq additive attention model using Bahdanau Attention
* USe a CNN encoder & RNN decoder to train the model for caption prediction
* Test the model on a test dataset

The model was adjusted and trained to fir our requiremets for the app.

### Sample Output
