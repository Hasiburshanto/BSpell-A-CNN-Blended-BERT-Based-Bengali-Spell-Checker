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

### These are defined in Variables.py file 
#word_char_size=10+2 
#sen_size=15 

### Data generator is required, as you can hardly load so many sample matrices at once in RAM 
class DataGenerator_new(keras.utils.Sequence):

  ## Replaces input parameter word characters by index obtained from tokenizer
  ## Suppose, word_char_size = 12 (with start and end marker)
  ## If length of a word is less then 10, then add zero to make the word size 10
  ## If a word length in greater than 10, then remove the extra charecters from end.
  def padding_word(self,dem):
    
    dem=dem.strip()
    word=[]
    word.append(self.Tokenizer.word_index.get('#'))
    
    if random()<=.45:
      isMask=1
    else:
      isMask=0

    ##Replace every word charecter by unique number
    for y in dem: 
      value_check=self.Tokenizer.word_index.get(y)  
      if (value_check==None):  # if it is a rarely used character, then assign the same index for all rare chars
        value_check=len(self.Tokenizer.word_index)+1
      if random()<=.25 and isMask==1:
        value_check=len(Tokenizer.word_index)+2
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
  def x_y_genarator_model_2(self,one_sentence):
      input1=[]
      count=0
      length=len(one_sentence) ##Full length of one sentence 

      for i in range(0,length): 
        j=one_sentence[i] ##pick every word (here word is unique number) form sentence.
        d=self.dict_Of_index_All_Words[j] 
        if ( random() > 0.20 ):
          input1.append(self.padding_word(d)) ## Word level paddding
        else:
          input1.append([len(self.Tokenizer.word_index)+2]*12) 
        count=count+1
      input11,c=self.padding_sentance(input1) ## Sentence level padding 
     
      output=[]
      count_output=0
      for i in range(c):
        count_output=count_output+1
        x=self.dict_Of_TOP_Words_index["PAD"]
        output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) )


      for i in one_sentence:
        count_output=count_output+1
        try: ###if 
          x=self.dict_Of_TOP_Words_index[dict_Of_index_All_Words[i]]
          output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) )
        except: ##recheck
          x=self.dict_Of_TOP_Words_index["UNK"]
          output.append(keras.utils.to_categorical(x, num_classes=self.n_classes) )
      output=output[0:15]
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
    return [X1], [y]
    
    
    