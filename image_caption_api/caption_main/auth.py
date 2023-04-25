'''
    IMPORTING
'''
from flask import Flask
app = Flask(__name__)



from flask import Blueprint , render_template , url_for , redirect , request ,session 
from werkzeug.utils import secure_filename
from flask import current_app
import wikipedia



# auth=Blueprint("auth",__name__)

'''
    THE URL WILL REDIRECT HERE FROM APP.PY
'''
@app.route('/' , methods=['GET','POST'])
def func():
    return render_template("detect.html")

import os
import pickle
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
# from matplotlib import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import add
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import keras.utils as image
vocab_size=1673
embedding_dim=200 
max_length=34
def create_model():
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model 







with open(r"E:\image_caption_fast_api\caption_main\all_files\wordtoix (1).pkl", "rb") as input_file:
    e = pickle.load(input_file)

with open(r"E:\image_caption_fast_api\caption_main\all_files\ixtoword (1).pkl", "rb") as input_file:
    f= pickle.load(input_file)
# print(e)
model_word= create_model()

# # Load the previously saved weights
# model.load_weights(latest)
model_word.load_weights("E:/image_caption_fast_api/caption_main/all_files/model_flickr8k (2).h5")

feature_extractor=tf.keras.models.load_model("E:/image_caption_fast_api/caption_main/all_files/fe.h5")
max_length=34
wordtoix=e
ixtoword=f



def greedySearch(feature_vector):
    print(wordtoix)
    in_text ='s###'
    for i in range(max_length):
        sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
        sequence = pad_sequences([sequence],maxlen = max_length)
        sequence=np.array(sequence)
        print(sequence)
        print(len(feature_vector))
        yhat = model_word.predict([feature_vector,sequence],verbose =0) # [(1,2048),(1,34)]
        yhat = np.argmax(yhat) # (1610,)
        word = ixtoword[yhat] 
        in_text += ' ' + word
        if word == 'e###' or word=='.':
            break
    final = in_text.split()
    final = final[1:-1] # list
    final = ' '.join(final) # convert list to string
    return final




def preprocess(image_path):
    img = image.load_img(image_path,target_size=  (299,299))
    x= image.img_to_array(img)
    x = np.expand_dims(x,axis =0) 
    x = preprocess_input(x)
    return x

def encode(image,feature_extractor):
    image = preprocess(image) # convert image model input
    feature_vector = feature_extractor.predict(image) # get the encoding vector for the image
    # feature_vector = np.ravel(feature_vector) # reshape (1,2048) to (2048,)
    return np.array(feature_vector) 

def finds(path):
    feature_vector = encode(path,feature_extractor)
    print(f'feature_vector: {feature_vector.shape}')
    final = greedySearch(feature_vector)
    return final

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        print(f.filename)
        result=finds(file_path)  
        return render_template("detect.html",task1=f.filename,tasks=result)
        




 


if __name__ == '__main__':
    app.run(debug=True)