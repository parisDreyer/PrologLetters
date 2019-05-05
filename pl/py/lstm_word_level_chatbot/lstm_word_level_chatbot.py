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
from keras.preprocessing import sequence
from nltk import FreqDist
import random
import util

os.environ['KMP_DUPLICATE_LIB_OK']='True' # https://github.com/openai/spinningup/issues/16
sys.stdout = stdout # return stdout to the terminal
sys.stderr = stderr

EXTRA_SYMBOLS = ['<PAD>', '<START>', '<UNK>', '<EOS>']
VOCAB_SIZE = 10000
loaded_model = util.load_from_disk(json_name = './py/lstm_word_level_chatbot/model.json', h5_name = './py/lstm_word_level_chatbot/model.h5')

def response(user_input_text, diversity = 1.2, response_length = 100):
    # formatting user input to vector encoding -- start again in the morning
    x, w21, i2w = format_words(user_input_text)
    x = batch_pad(x, add_eos=True)
    b = random.choice(x)
    seed = b[0, :]
    gen = generate_seq(seed, response_length, temperature=diversity)
    return util.decode(i2w, gen[response_length:])

def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str

def format_words(input_text):
    """
    Loads sentences (or other natural language sequences) from an input-string. Assumes a single sequence per line.

    :param source: Text file to read from
    VOCAB_SIZE: Maximum number of words to retain. If there are more unique words than this, the most frequent
        "vocab_size" words are used, and the rest are replaced by the <UNK> symbol

    :return: (1) A list of lists of integers representing the encoded sentences, (3) a dict from strings to ints
        representing the mapping from words to indices (2) a list of strings representing the mapping from indices to
        words.
    """
    x_data = EXTRA_SYMBOLS[1] + "\n" + input_text + "\n" + EXTRA_SYMBOLS[3]
    x = [text_to_word_sequence(x) for x in x_data.split('\n') if len(x) > 0] # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    dist = FreqDist(np.hstack(x))
    x_vocab = dist.most_common(VOCAB_SIZE - len(EXTRA_SYMBOLS))
    i2w = [word[0] for word in x_vocab]            # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    i2w = EXTRA_SYMBOLS + i2w                      # Adding the word "ZERO" to the beginning of the array
    w2i = {word:ix for ix, word in enumerate(i2w)} # Creating the word-to-index dictionary from the array created above
    for i, sentence in enumerate(x):               # Converting each word to its index value
        for j, word in enumerate(sentence):
            if word in w2i:
                x[i][j] = w2i[word]
            else:
                x[i][j] = w2i['<UNK>']
    return x, w2i, i2w

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

def batch_pad(x, batch_size = 2, min_length=3, add_eos=False, extra_padding=0):
    """
    Takes a list of integer sequences, sorts them by lengths and pads them so that sentences in each batch have the
    same length.

    :param x:
    :return: A list of tensors containing equal-length sequences padded to the length of the longest sequence in the batch
    """
    x = sorted(x, key=lambda l : len(l))
    if add_eos:
        eos = EXTRA_SYMBOLS.index('<EOS>')
        x = [sent + [eos,] for sent in x]
    batches = []
    start = 0
    while start < len(x):
        end = start + batch_size
        if end > len(x):
            end = len(x)
        batch = x[start:end]
        mlen = max([len(l) + extra_padding for l in batch])
        if mlen >= min_length:
            batch = sequence.pad_sequences(batch, maxlen=mlen, dtype='int32', padding='post', truncating='post')
            batches.append(batch)
        start += batch_size
    return batches
