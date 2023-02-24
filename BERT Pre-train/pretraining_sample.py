import numpy as geek
from keras.preprocessing.text import Tokenizer
import pickle
from random import random
from Variables import *

"""
****  Input: Dataset (a text file containing a full sentence per line) and the output files obtained from
             running data.py file.
****  Output: Training and validation samples.



=========================================================================================================

path :: Dataset (a text file containing a full sentence per line) path
dic  :: The folder path where all data.py output pickle files have been saved  

"""
#This variable is defined in Variables.py file 
#sentence_topWord_percentage=85
path="/content/bangla 3k.txt"
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""

# Loading the necessary dictionary mappings 
with open(dic+'dict_Of_All_Words_Index.pickle', 'rb') as handle:
    dict_Of_All_Words_Index = pickle.load(handle)

with open(dic+'dict_Of_TOP_Words_index.pickle', 'rb') as handle:
    dict_Of_TOP_Words_index = pickle.load(handle)

with open(dic+'dict_Of_index_All_Words.pickle', 'rb') as handle:
    dict_Of_index_All_Words = pickle.load(handle)


## Load all sentences into "Sentences" variable
Sentences=[]
with open(path,'r',encoding = 'utf-8') as f:
  for line in f:
      Sentences.append(line)

## Replaces each word of the sentences with corresponding word index 
unique_Sentance_number_represent=[] 
for i in Sentences:
  x=i.strip()
  x=x.split()
  flag=0 
  d=[]
  for j in x:
    try:
      d.append(dict_Of_All_Words_Index[j])
    except:
      flag=1
  if len(d)>1 and flag==0:
    unique_Sentance_number_represent.append(d)

print(len(unique_Sentance_number_represent))

## Keep only those sentences which have at least sentence_topWord_percentage portion of top k words  
temp_dataset=[]

for i in unique_Sentance_number_represent:
      UNK,WORD=0,0
      for j in i:
        y=dict_Of_index_All_Words[j] 
        if y not in dict_Of_TOP_Words_index:
          UNK=UNK+1
        else: 
          WORD=WORD+1
      
      if  (WORD/len(i))*100  >=sentence_topWord_percentage:
        temp_dataset.append(i)

print("New Size of dataset:: ",len(temp_dataset),"Old Size of dataset:: ",len(unique_Sentance_number_represent) )



##Saved Training and validation Data.
with open(dic+'pre-Train.data', 'wb') as filehandle:
    pickle.dump(temp_dataset, filehandle)
