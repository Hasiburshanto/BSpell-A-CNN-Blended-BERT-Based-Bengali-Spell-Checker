import pickle
import numpy as np
from random import random
import keras
from keras.preprocessing.text import Tokenizer
from random import random, choice
from random import *
import math 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 
import numpy as np
import pandas as pd
from Variables import  *
import numpy as geek


##dic:: folder path of where you want to save your trained model 
dic=""
epoch_number=1
Batch_size=64

##root:: Saved model path
root=""
## Already created these files and saved.
with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)
with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words = pickle.load(handle)
with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)
with open(dic+'Train.data', 'rb') as handle:
    Train_data = pickle.load(handle)
with open(dic+'Valid.data', 'rb') as handle:
    Valid_data = pickle.load(handle)
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 

with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)

Train_data=np.array(Train_data) 
train_data_length=len(Train_data)

Valid_data=np.array(Valid_data)
valid_data_length=len(Valid_data)


### Data generator is required, as you can hardly load so many sample matrices at once in RAM 
class DataGenerator_new(keras.utils.Sequence):

  ## Replaces input parameter word characters by index obtained from tokenizer
  ## Suppose, word_char_size = 12 (with start and end marker)
  ## If length of a word is less then 10, then add zero to make the word size 10
  ## If a word length in greater than 10, then remove the extra charecters from end.
  def padding_word(self,dem):

    word=[]
    word.append(self.Tokenizer.word_index.get('#'))    
    ##Replace every word charecter by unique number
    for y in dem: 
      value_check=self.Tokenizer.word_index.get(y)  
      if(value_check==None):  # if it is a rarely used character, then assign the same index for all rare chars
        value_check=len(self.Tokenizer.word_index)+1
      word.append(value_check)
    word.append(self.Tokenizer.word_index.get('$')) 

    flag=0
    out=[]

    if (len(word)<self.word_char_size): ##12 because of two extra charecter (end-$ and begging-# charecter)
      word_len=len(word)
      dif=self.word_char_size-word_len
      flag=1
      for c in range(0,math.ceil(dif/2)):  # adding 0's in the beginning
        out.append(0)
      for c in word:  # the valid characters
        out.append(c)
      for c in range(0,math.floor(dif/2)): # adding 0's at the end 
        out.append(0)
 
    elif (len(word)>=self.word_char_size and flag==0): # stripping the word if word is too large
      out=word[0:self.word_char_size]  
      out[11]=self.Tokenizer.word_index.get('$')
    return out

  ## padding a sentence. sen_size is the maximum sentence length (Last word is the output). 
  ## Suppose, sen_size = 15. If a sentance size is less than 14, then add zero padded words 
  ## to make that sentance size 14. If a sentance size is greater than 14, then remove the extra 
  ## words from the end. 
  ## It also return how many word is padded in one sentence.
  
  def padding_sentance(self,dem):
    pad_count=0
    if(len(dem)<self.sen_size):  ## sen_size-1 because last one is target.
      x=len(dem)
      pad_sen=[]
      dif=self.sen_size-x
      for i in range(0,dif):
        pad_count=pad_count+1
        pad_sen.append([0]*self.word_char_size) 
      for c in dem:
        pad_sen.append(c)
    else:
      pad_sen=dem[0:self.sen_size] 
    return pad_sen,pad_count

  # Input parameter is a sample. Function outputs: model input matrix, mask input and word prediction output 
  def x_y_genarator_model_2(self,correct_vs_error_sentence):
      input1=[]
      count=0
      erorr_sentence=correct_vs_error_sentence[1] ##Because Input is error sentence
      correct_sentence=correct_vs_error_sentence[0]
      length=len(erorr_sentence) ##Full length of one correct sentence 

      for i in range(0,length): 
        d=erorr_sentence[i]
        input1.append(self.padding_word(d)) ## Word level paddding

        count=count+1
      input11,c=self.padding_sentance(input1) ## Sentence level padding 
     
      output=[]
      count_output=0
      for i in range(c): ## C is the number, which repersent how many word is padded.
        count_output=count_output+1
        x=self.dict_Of_TOP_Words_index["PAD"]
        output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) ) ##Append pad word hot vector to output set.


      for i in correct_sentence:
        count_output=count_output+1
        try: ###if 
          x=self.dict_Of_TOP_Words_index[i]
          output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) )
        except: ##If word isn't in top word dictonary then it must be "UNK" word.
          x=self.dict_Of_TOP_Words_index["UNK"]
          output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) )
      output=output[0:15]
      #print(input11.shape,output.shape)
      return input11,output

  ### Input parameters: samples, sample no., Batch size, top word no.,
    #                   index to unique word mapping, unique word to index mapping,
    #                   index to top word mapping, top word to index mapping, character fitted tokenizer
  def __init__(self, samples, de1,batch_size, n_classes,
               dict_Of_index_All_Words,dict_Of_All_Words_Index,
               dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
               Tokenizer,sen_size,word_char_size,CNN_Vec_size):
    # Initializing variables
    self.samples = samples
    self.batch_size = batch_size
    self.dim1 = de1 
    self.n_classes = n_classes
    self.dict_Of_index_All_Words=dict_Of_index_All_Words
    self.dict_Of_All_Words_Index=dict_Of_All_Words_Index
    self.dict_Of_index_Top_Words=dict_Of_index_Top_Words
    self.dict_Of_TOP_Words_index=dict_Of_TOP_Words_index
    self.Tokenizer=Tokenizer
    self.sen_size=sen_size
    self.word_char_size=word_char_size
    self.CNN_Vec_size=CNN_Vec_size
    self.on_epoch_end()

  def __len__(self):
    #Denotes the number of batches per epoch'
    return int(np.floor(len(self.samples) / self.batch_size))

  def __getitem__(self, index):  # sends data per batch
    # Generate one batch of data
    X, y = self.__data_generation(index) # index denotes batch no. 
    return X, y

  def on_epoch_end(self):
    # shuffling the samples at the end of each epoch 
    ind = np.arange(self.dim1)
    np.random.shuffle(ind)
    self.samples = self.samples[ind]
  def __data_generation(self, index):
    X1 = []
    y = []
  
    for i in range(index*self.batch_size, (index+1)*self.batch_size):
      one_se= self.samples[i,]   

      a,d=self.x_y_genarator_model_2(one_se) 
      #print("okkk")
      ### a is input sentence, e is mask input, d is true target.
      X1.append(a)
      y.append(d)
    
    X1=np.asarray(X1)
    y=np.asarray(y)   
    ## shape [Batch_size,1,Top_word_size] reshaped to [Batch_size,Top_word_size]
    return [X1], [y,y]
    
    
top_word_size=len(top_k_word)

training_generator = DataGenerator_new(Train_data,train_data_length, Batch_size,top_word_size,
                                                dict_Of_index_All_Words,dict_Of_All_Words_Index,
                                                dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
                                                Tokenizer,sen_size,word_char_size,CNN_Vec_size)

valid_generator = DataGenerator_new(Valid_data,valid_data_length, Batch_size,top_word_size,
                                                dict_Of_index_All_Words,dict_Of_All_Words_Index,
                                                dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
                                                Tokenizer,sen_size,word_char_size,CNN_Vec_size)

print(top_word_size)
model = keras.models.load_model(root+"Fine_Tune_Model")
model.fit_generator(training_generator,validation_data=valid_generator, epochs=epoch_number, verbose=1,
                    workers=6)
model.save(root+"Fine_Tune_Model")