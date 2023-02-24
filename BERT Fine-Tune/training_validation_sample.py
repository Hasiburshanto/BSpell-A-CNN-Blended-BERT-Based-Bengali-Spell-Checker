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




#This variable is defined in Variables.py file 
#sentence_topWord_percentage=85

path="/content/sample data.txt" ##Fine tune correct dataset
error_path="/content/error sample data.txt" ##Error dataset

dic="/content/drive/My Drive/AAAI Code With Dataset/" ## "dic" is the path where dictionaries are saved
dic=""

# Loading the necessary dictionary mappings 
with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)
with open(dic+'dict_Of_index_Top_Words.pickle', 'rb') as handle:
    dict_Of_index_Top_Words = pickle.load(handle)
with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)
with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 
with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)

## Load all sentences into "Sentences" variable
Sentences=[]
with open(path,'r',encoding = 'utf-8') as f:
  for line in f:
      Sentences.append(line)


## Load all sentences into "Sentences" variable
Error_Sentences=[]
with open(error_path,'r',encoding = 'utf-8') as f:
  for line in f:
      Error_Sentences.append(line)


## Replaces each word of the sentences with corresponding word index 
Error_Sentence_Vs_Sentance=[] 
for j in range(len(Sentences)):
  i=Sentences[j] ##correct sentence 
  y=i.strip() ##Remove End char '\n' from sentence.
  y=y.split()

  k=Error_Sentences[j]
  x=k.strip() 
  x=x.split()
  
  Error_Sentence_Vs_Sentance.append([y,x])

print(len(Error_Sentence_Vs_Sentance))

## Keep only those sentences which have at least sentence_topWord_percentage portion of top k words  
temp_dataset=[]

for i in Error_Sentence_Vs_Sentance:
      UNK,WORD=0,0
      for j in i[0]:
        y=j
        if y not in dict_Of_TOP_Words_index:
          UNK=UNK+1
        else: 
          WORD=WORD+1
      
      if  (WORD/len(i))*100  >=sentence_topWord_percentage:
        temp_dataset.append(i)

print("New Size of dataset:: ",len(temp_dataset),"Old Size of dataset:: ",len(Error_Sentence_Vs_Sentance) )


## Make samples and Split into Training and validation set.
# Each sample contains some input words and the next word as output 

Train_data=[]
Valid_data=[]
for i in temp_dataset:
  train=0
  if (random()<=.90):  ## we are performing a 90-10 split. You can change this 
        
    train=1

  if train==1:
      Train_data.append(i)
  else:
      Valid_data.append(i)
      
  
print("Training sample size :: ",len(Train_data))  
print("Validation sample size :: ",len(Valid_data))  

##Saved Training and validation Data.
with open(dic+'Train.data', 'wb') as filehandle:
    pickle.dump(Train_data, filehandle)

with open(dic+'Valid.data', 'wb') as filehandle:
    pickle.dump(Valid_data, filehandle)