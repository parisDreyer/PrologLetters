from __future__ import print_function
# silence keras loading messages -- got from: https://github.com/keras-team/keras/issues/1406
import io
import os
import sys
import tensorflow as tf
import numpy as np
stdout = sys.stdout
stderr = sys.stderr
sys.stdout = open('/dev/null', 'w')
sys.stderr = open('/dev/null', 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow output
np.seterr(divide='ignore') # silence divide by zero warning
# =================================
from keras.preprocessing.text import text_to_word_sequence
import keras.utils
import keras.backend as K
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import random
import util

os.environ['KMP_DUPLICATE_LIB_OK']='True' # https://github.com/openai/spinningup/issues/16
sys.stdout = stdout # return stdout to the terminal
sys.stderr = stderr
loaded_model = util.load_from_disk(json_name = './py/lstm_word_level_chatbot/model.json', h5_name = './py/lstm_word_level_chatbot/model.h5')

def response(user_input_text, diversity = 1.2, response_length = 100):
    # formatting user input to vector encoding -- start again in the morning
    x = util.format_words(user_input_text)
    x = util.batch_pad(x, add_eos=True)
    b = random.choice(x)
    seed = b[0, :]
    gen = generate_seq(seed, response_length, temperature=diversity)
    return util.decode(gen[response_length:])

def to_categorical(batch, num_classes): # Converts a batch of length-padded integer sequences to a one-hot encoded sequence
    b, l = batch.shape
    out = np.zeros((b, l, num_classes))
    for i in range(b):
        seq = batch[0, :]
        out[i, :, :] = keras.utils.to_categorical(seq, num_classes=num_classes)
    return out

def generate_seq(seed, size, temperature=1.0): # :return: A list of integers representing a samples sentence
    seed = np.insert(seed, 0, 1)
    ls = seed.shape[0]
    tokens = np.concatenate([seed, np.zeros(size - ls)])
    for i in range(ls, size):
        probs = loaded_model.predict(tokens[None,:])
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature) # Extract the i-th probability vector and sample an index from it
        tokens[i] = next_token
    return [int(t) for t in tokens]

