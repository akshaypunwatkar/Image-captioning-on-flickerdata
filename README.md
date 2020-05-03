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
* Implement a Bahdanau Attention based recurrent neural network
* Use a CNN encoder & RNN decoder to train the model for caption prediction
* Test the model on a test dataset

The model was adjusted and trained to fit our requiremets for the app.

### Sample Output

Below is an example of the caption generated by our model for the given input image:
![Sample Output](https://github.com/akshaypunwatkar/Image-captioning-on-flickerdata/blob/master/demo_sample_output.PNG)

## App Deployment

This app was deployed ...

## Load Testing

We used the Locust Software on the Google Kubernetes engine to check our app for load testing. For this, we followed the step-by-step tutorial given [here](https://cloud.google.com/solutions/distributed-load-testing-using-gke).

Further details on the performance can be seen in the demo video linked below.

## Demo Video

Link to the video: [here](https://youtu.be/zaQ3NOj1oJo)
