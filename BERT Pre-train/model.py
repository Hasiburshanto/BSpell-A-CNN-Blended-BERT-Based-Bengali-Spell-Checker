# Building and saving our proposed BSpell model with randomly initialized weights (no auxiliary loss)
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
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)
    
character_size=len(Tokenizer.word_index)+3
unique_words=len(top_k_word)
top_word_size=unique_words

### Building the vocabulary learner sub-model 
visible1 = layers.Input(shape=(sen_size, word_char_size,))
embedding_layer_1=Embedding(character_size,
              output_dim=char_vec_size)(visible1) 
 
x = TimeDistributed(Conv1D(64,2,padding='same',strides=1))(embedding_layer_1)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(64,3,padding='same', strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(128,3,padding='same',strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(128,3,padding='same',strides=1))(x)
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(Conv1D(CNN_Vec_size,4,padding='same',strides=1))(x) 
x = TimeDistributed( BatchNormalization())(x)
x = TimeDistributed(Activation("relu"))(x)

x = TimeDistributed(GlobalMaxPooling1D())(x) ## get CNNvecs for all input words

num_heads = 8
ff_dim = 128
latent_dim = 256
embed_dim = 512
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim # 512
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads # 64
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True) # (64, 8, 20, 20)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32) # 64
        scaled_score = score / tf.math.sqrt(dim_key) # score/ 8 -> (64, 8, 20, 20)
        weights = tf.nn.softmax(scaled_score, axis=-1) # (64, 8, 20, 20) -> softmax on last dimension
        output = tf.matmul(weights, value) # (64, 8, 20, 64)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim)) # (64, 20, 8, 64)
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (64, 8, 20, 64)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] # 64
        query = self.query_dense(inputs)  # (64, 20, 512)
        key = self.key_dense(inputs)  # (64, 20, 512)
        value = self.value_dense(inputs)  # (64, 20, 512)
        query = self.separate_heads(
            query, batch_size
        )  # (64, 8, 20, 64)
        key = self.separate_heads(
            key, batch_size
        )  # (64, 8, 20, 64)
        value = self.separate_heads(
            value, batch_size
        )  # (64, 8, 20, 64)
        attention = self.attention(query, key, value) # (64, 8, 20, 64)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (64, 20, 8, 64)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (64, 20, 512)
        output = self.combine_heads( # A dense layer
            concat_attention
        )  # (64, 20, 512)
        return output


class PositionEmbedding(layers.Layer):
    def __init__(self, length, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=length, output_dim=embed_dim)
        self.maxlen = length

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        return (x + positions)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] # No activation means linear activation
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)  # (64, 20, 512)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1) # (64, 20, 512)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # (64, 20, 512)        

def softMaxAxis1(x):
  return softmax(x ,axis=-1)

x = TimeDistributed(Dense(512, activation='relu'))(x)
transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block1(x) 
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block2(x) 
transformer_block3 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block3(x) 
transformer_block4 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block4(x)
transformer_block5 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block5(x) 
transformer_block6 = TransformerBlock(embed_dim, num_heads, ff_dim, Dropout_Rate)
x = transformer_block6(x) 

output = Dense(unique_words, activation=softMaxAxis1)(x)
model = keras.Model(inputs=[visible1], outputs=output)

opt = keras.optimizers.SGD(learning_rate=learning_rate,clipnorm=clipnorm)
model.compile(optimizer= opt, loss= 'categorical_crossentropy', metrics='accuracy')

### Viewing model diagram
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.save(dic+"PreTrain_Model")