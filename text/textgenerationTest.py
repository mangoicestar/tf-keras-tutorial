#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:47:24 2017

@author: jayhsu
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['THEANO_FLAGS']="floatX=float64,device=cpu"
#os.environ['THEANO_FLAGS']="floatX=float32,device=cuda"

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
import keras.backend as K
import numpy as np
import random
import sys


text = open("sgyy_all.txt").read().lower().replace("\n\n", "\n").replace("\n\n", "\n")
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 30
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))


#Vectorization
X = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences),), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_indices[char] 
    y[i] = char_indices[next_chars[i]]
    
    
#The model
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
        return K.mean(K.in_top_k(y_pred, K.max(y_true, axis=-1), k))
    
    
model = Sequential()
model.add(Embedding(len(chars), 128))
model.add(LSTM(512, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
model.fit(X, y, batch_size=12800, epochs=1)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


start_index = random.randint(0, len(text) - maxlen - 1)
generated = ''
sentence = text[start_index: start_index + maxlen]   
generated += sentence   
diversity=0.5
for i in range(100):
    x = np.zeros((1, maxlen))
    for t, char in enumerate(sentence):
        x[0, t] = char_indices[char]

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char      
print(generated)


