import numpy as np
from scipy.misc import logsumexp
from collections import defaultdict, Counter, OrderedDict

from keras.models import model_from_json
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Layer
import keras.utils

def sample_logits(preds, temperature=1.0): # "Sample an index from a logit vector."
    preds = np.asarray(preds).astype('float64')
    if temperature == 0.0:
        return np.argmax(preds)
    preds = preds / temperature
    preds = preds - logsumexp(preds)
    return np.random.choice(len(preds), 1, p=np.exp(preds))

def load_from_disk(json_name = 'model.json', h5_name = 'model.h5'):
    # load json and create model
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_name)
    print("Loaded model from disk")
    return loaded_model

def decode(i2w, seq):
    return ' '.join(i2w[id] for id in seq)
