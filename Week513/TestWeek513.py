from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from Week513.grammar import *
from Week513.qa import *
from Week513.preprocess import *
from Week513.music_utils import *
from Week513.data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

#IPython.display.Audio('data/30s_seq.mp3')
X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

