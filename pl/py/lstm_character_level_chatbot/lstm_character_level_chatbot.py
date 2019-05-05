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
# from keras.callbacks import LambdaCallback
# from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import random
import util
os.environ['KMP_DUPLICATE_LIB_OK']='True' # https://github.com/openai/spinningup/issues/16
sys.stdout = stdout # return stdout to the terminal
sys.stderr = stderr

PATH = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
TEXT = ''
with io.open(PATH, encoding='utf-8') as f:
    TEXT = f.read().lower()

CHARS = sorted(list(set(TEXT)))
CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))
INDICES_CHARS = dict((i, c) for i, c in enumerate(CHARS))
MAX_LENGTH = 40
loaded_model = util.load_from_disk(json_name='./py/lstm_character_level_chatbot/model.json', h5_name='./py/lstm_character_level_chatbot/model.h5')

def response(user_input_text, diversity = 1.2, response_length = 100): # Prints generated text.
        answer = ''
        sentence = user_input_text
        if len(sentence) >= MAX_LENGTH: # prevent errors in the expected input vector
            sentence = sentence[:MAX_LENGTH]
        for i in range(response_length):
            x_pred = np.zeros((1, MAX_LENGTH, len(CHARS)))
            for t, char in enumerate(sentence):
                x_pred[0, t, CHAR_INDICES[char]] = 1.
            next_char = INDICES_CHARS[sample(loaded_model.predict(x_pred, verbose=0)[0], diversity)]
            answer += next_char
            sentence = sentence[1:] + next_char
        return answer

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
