from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # https://github.com/openai/spinningup/issues/16

path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40

# load json and create model
json_file = open('./py/lstm_chatbot/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./py/lstm_chatbot/model.h5")

def lstm_chatbot_response(user_input_text, diversity = 1.2, response_length = 100): # Prints generated text.
        response = ''
        sentence = user_input_text
        if len(sentence) >= maxlen: # prevent errors in the expected input vector
            sentence = sentence[:maxlen]
        for i in range(response_length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
            next_char = indices_char[sample(loaded_model.predict(x_pred, verbose=0)[0], diversity)]
            response += next_char
            sentence = sentence[1:] + next_char
        return response

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
