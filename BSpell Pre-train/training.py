
import genaretor ##Class

import pickle
import numpy as np
from tensorflow import keras
from Variables import *
##dic:: folder path of where you want to save your trained model 
dic="/content/drive/My Drive/AAAI Code With Dataset/"
dic=""
epoch_number=1
Batch_size=64
##root:: Saved model path
root="/content/drive/My Drive/AAAI Code With Dataset/"
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
with open(dic+'pre-Train.data', 'rb') as handle:
    pre_Train = pickle.load(handle)


with open(dic+'tokenizer.pickle', 'rb') as handle:
    Tokenizer = pickle.load(handle) 

with open(dic+'top_k_word.pickle', 'rb') as handle:
    top_k_word = pickle.load(handle)

pre_Train=np.array(pre_Train) 
pre_Train_length=len(pre_Train)


top_word_size=len(top_k_word)

training_generator = genaretor.DataGenerator_new(pre_Train,pre_Train_length, Batch_size,top_word_size,
                                                dict_Of_index_All_Words,dict_Of_All_Words_Index,
                                                dict_Of_index_Top_Words,dict_Of_TOP_Words_index,
                                                Tokenizer,sen_size,word_char_size,CNN_Vec_size)

print(top_word_size)


model = keras.models.load_model(root+"PreTrain_Model")

model.fit_generator(training_generator, epochs=epoch_number, verbose=1,
                    workers=6)

model.save(root+"PreTrain_Model")