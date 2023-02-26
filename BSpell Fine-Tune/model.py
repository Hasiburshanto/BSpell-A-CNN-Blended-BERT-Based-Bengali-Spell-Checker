# Loading pretrained BSpell model and adding auxiliary loss 
import pickle
import numpy as np
from random import random
import keras
from keras.preprocessing.text import Tokenizer
import math 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Multiply, Dense, Dropout, Activation, Flatten, Bidirectional, LSTM, Activation, dot, Embedding,BatchNormalization
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from Variables import  *
from tensorflow.keras.activations import softmax

## The folder path where all data.py output pickle files have been saved and also where you want to save your model
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""
root=""
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)
    
character_size=len(Tokenizer.word_index)+3
unique_words=len(top_k_word)
top_word_size=unique_words
model = keras.models.load_model(root+"PreTrain_Model")

### Building the vocabulary learner sub-model 
visible1 = layers.Input(shape=(sen_size, word_char_size,))

pre_Bert_layers = model.layers[1:19]
pre_Bert_layers2 = model.layers[19:-1]

x=visible1

for i in range(0, len(pre_Bert_layers)):
  x = pre_Bert_layers[i](x)
outputs1 = Dense(unique_words, activation= softMaxAxis1, name="output1")(x)
for i in range(0, len(pre_Bert_layers2)):
  x = pre_Bert_layers2[i](x)

outputs2 = Dense(unique_words, activation= softMaxAxis1, name="output2")(x)
model2 = keras.Model(inputs= visible1, outputs=[outputs1, outputs2])


opt = keras.optimizers.SGD(learning_rate=learning_rate,clipnorm=clipnorm)

model2.compile(optimizer=opt, 
              loss={
                  'output1': 'categorical_crossentropy', 
                  'output2': 'categorical_crossentropy'
                  },
              loss_weights={
                  'output1': 0.3, 
                  'output2': 1 
                  },
              metrics={
                  'output1': [ 'accuracy'], 
                  'output2': [ 'accuracy']
                  }) # 
model2.save(dic+"Fine_Tune_Model")