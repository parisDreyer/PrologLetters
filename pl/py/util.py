import numpy as np
from scipy.misc import logsumexp
from collections import defaultdict, Counter, OrderedDict
from keras.models import model_from_json
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Layer
import keras.utils
from keras.preprocessing import sequence
from nltk import FreqDist

# VOCAB_SIZE: Maximum number of words to retain. If there are more unique words than this, the most frequent
# "vocab_size" words are used, and the rest are replaced by the < UNK > symbol
VOCAB_SIZE = 10000
EXTRA_SYMBOLS = ['<PAD>', '<START>', '<UNK>', '<EOS>']
def load_words(source):
    # Reading raw text from source and destination files
    f = open(source, 'r')
    x_data = f.read()
    f.close()
    # Splitting raw text into array of sequences
    x = [text_to_word_sequence(x) for x in x_data.split('\n') if len(x) > 0]
    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    dist = FreqDist(np.hstack(x))
    x_vocab = dist.most_common(VOCAB_SIZE - len(EXTRA_SYMBOLS))
    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    i2w = [word[0] for word in x_vocab]
    # Adding the word "ZERO" to the beginning of the array
    i2w = EXTRA_SYMBOLS + i2w
    # Creating the word-to-index dictionary from the array created above
    w2i = {word: ix for ix, word in enumerate(i2w)}
    # Converting each word to its index value
    for i, sentence in enumerate(x):
        for j, word in enumerate(sentence):
            if word in w2i:
                x[i][j] = w2i[word]
            else:
                x[i][j] = w2i['<UNK>']
    return x, w2i, i2w
X, WORD_TO_INDEX, INDEX_TO_WORD = load_words('./py/word_dataset/wikisimple.txt')

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
    return loaded_model

def decode(seq):
    return ' '.join(INDEX_TO_WORD[id] for id in seq)


def format_words(input_text):
    # Loads sentences (or other natural language sequences) from an input-string. Assumes a single sequence per line.
    # :param source: Text file to read from
    # :return: (1) A list of lists of integers representing the encoded sentences, (3) a dict from strings to ints
    #     representing the mapping from words to indices (2) a list of strings representing the mapping from indices to words.

    x_data = EXTRA_SYMBOLS[1] + "\n" + input_text + "\n" + EXTRA_SYMBOLS[3]
    # Creating the vocabulary set with the most common words (leaving room for PAD, START, UNK)
    x = [text_to_word_sequence(x) for x in x_data.split('\n') if len(x) > 0]
    # dist = FreqDist(np.hstack(x))
    # x_vocab = dist.most_common(VOCAB_SIZE - len(EXTRA_SYMBOLS))
    # # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    # i2w = [word[0] for word in x_vocab]
    # # Adding the word "ZERO" to the beginning of the array
    # i2w = EXTRA_SYMBOLS + i2w
    # # Creating the word-to-index dictionary from the array created above
    # w2i = {word: ix for ix, word in enumerate(i2w)}
    # Converting each word to its index value
    for i, sentence in enumerate(x):
        for j, word in enumerate(sentence):
            if word in WORD_TO_INDEX:
                x[i][j] = WORD_TO_INDEX[word]
            else:
                x[i][j] = WORD_TO_INDEX['<UNK>']
    return x


def batch_pad(x, batch_size=2, min_length=3, add_eos=False, extra_padding=0):
    # Takes a list of integer sequences, sorts them by lengths and pads them so that sentences in each batch have the same length.
    # :return: A list of tensors containing equal-length sequences padded to the length of the longest sequence in the batch
    
    x = sorted(x, key=lambda l: len(l))
    if add_eos:
        eos = EXTRA_SYMBOLS.index('<EOS>')
        x = [sent + [eos, ] for sent in x]
    batches = []
    start = 0
    while start < len(x):
        end = start + batch_size
        if end > len(x):
            end = len(x)
        batch = x[start:end]
        mlen = max([len(l) + extra_padding for l in batch])
        if mlen >= min_length:
            batch = sequence.pad_sequences(
                batch, maxlen=mlen, dtype='int32', padding='post', truncating='post')
            batches.append(batch)
        start += batch_size
    return batches

# def idx2word(idx, i2w, pad_idx):
#     sent_str = [str()]*len(idx)
#     for i, sent in enumerate(idx):
#         for word_id in sent:
#             if word_id == pad_idx:
#                 break
#             sent_str[i] += i2w[str(word_id.item())] + " "
#         sent_str[i] = sent_str[i].strip()
#     return sent_str

def format_bot_output(text):
    if len(text) == 0:
        return text
    ascii_parsed = "".join([i if ord(i) < 128 else "" for i in text]).split(" ")
    return " ".join([translate_word_from_sym(w) for w in ascii_parsed])

def translate_word_from_sym(word):
    if word == EXTRA_SYMBOLS[0]:
        return "   "
    elif word == EXTRA_SYMBOLS[1]:
        return ""
    elif word == EXTRA_SYMBOLS[2]:
        return "_"
    elif word == EXTRA_SYMBOLS[3]:
        return "."
    else:
        return word
