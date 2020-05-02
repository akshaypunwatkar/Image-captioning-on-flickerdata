from flask import Flask,request, jsonify, render_template
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import re
import numpy as np
import os
import time
import json
import requests
from glob import glob
from PIL import Image
import pickle

app = Flask(__name__)


image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
  	def __init__(self, embedding_dim, units, vocab_size):
	    super(RNN_Decoder, self).__init__()
	    self.units = units

	    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
	    self.gru = tf.keras.layers.GRU(self.units,
	                                   return_sequences=True,
	                                   return_state=True,
	                                   recurrent_initializer='glorot_uniform')
	    self.fc1 = tf.keras.layers.Dense(self.units)
	    self.fc2 = tf.keras.layers.Dense(vocab_size)

	    self.attention = BahdanauAttention(self.units)

  	def call(self, x, features, hidden):
	    context_vector, attention_weights = self.attention(features, hidden)
	    x = self.embedding(x)
	    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
	    output, state = self.gru(x)
	    x = self.fc1(output)
	    x = tf.reshape(x, (-1, x.shape[2]))
	    x = self.fc2(x)

	    return x, state, attention_weights

  	def reset_state(self, batch_size):
  		return tf.zeros((batch_size, self.units))


def load_image(image_path,is_url=False):
    if is_url:
        img = tf.image.decode_jpeg(requests.get(image_path).content, channels=3)
    else:
        img = tf.io.read_file(image_path)    
        img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def evaluate(image, encoder1, decoder1,url_flag=False):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder1.reset_state(batch_size=1)

    #loading image
    temp_input = tf.expand_dims(load_image(image,url_flag)[0], 0)

    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder1(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer1.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder1(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer1.index_word[predicted_id])

        if tokenizer1.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

embedding_dim = 256
units = 512
top_k = 5000
max_length = 52
vocab_size = top_k + 1
features_shape = 2048
attention_features_shape = 64   	


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

test_tensor_img = tf.convert_to_tensor(np.ones((64, 64, 2048)))
target = tf.convert_to_tensor(np.ones((64, 49)))

encoder1 = CNN_Encoder(embedding_dim)
encoder1(test_tensor_img)
encoder1.load_weights('weights/encoder.h5')
decoder1 = RNN_Decoder(embedding_dim, units, vocab_size)
hidden = decoder1.reset_state(batch_size=test_tensor_img.shape[0])
dec_input = tf.expand_dims([tokenizer1.word_index['<start>']] * target.shape[0], 1)
features = encoder1(test_tensor_img)
decoder1(dec_input, features, hidden)
decoder1.load_weights('weights/decoder.h5')

#Default Page
@app.route('/')
def hello():
    return render_template("index.html")

#GUI predict
@app.route('/results')
def predict_gui():
    return "something"


#REST API
@app.route('/predict',methods=['POST','GET'])    
def predict_rest():
    data = {"success": False}

    if request.method in ["POST","GET"]:
        if request.form.get("filepath"):
            path = request.form.get("filepath")
            is_url = False 
            #checking if the input path is a URL 
            if path.startswith("http"):
                is_url = True
            #making the prediction for the caption    
            result, attention_plot = evaluate(path, encoder1, decoder1,is_url)
            data["caption"] = " ".join(result[:-1])
            data['filepath'] = path
            data["success"] = True

    return jsonify(data) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=True,threaded=False)

#curl -X GET -F filepath=images/COCO_train2014_000000050592.jpg 'http://localhost:8088/predict'
#curl -X POST -F filepath=images/COCO_train2014_000000050592.jpg 'http://localhost:8088/predict'
#curl -X GET -F filepath=https://media.stadiumtalk.com/51/78/5178471c78244562a6fa79e0e14d7a32.jpg 'http://localhost:8088/predict'





