import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import \
    Dense, LSTM, Embedding, TimeDistributed, Bidirectional, SpatialDropout1D, GRU, Input
from keras.models import Model
from tensorflow.python.client import device_lib

from tensorboardX import SummaryWriter

from keras.utils import multi_gpu_model

from tqdm import tqdm
import math, sys, os, random
import numpy as np
import util

encoder, decoder = load_encoder_and_decoder()

def generate_seq(model, z, size = 60, seed = np.ones(1), temperature=1.0):
    """ :param model:Model, :param z: The latent vector from which to generate, :param size:, :param lstm_layer:, :param seed:, :param temperature:
    :return: A list of integers representing a sentence.
    """
    ls = seed.shape[0]
    tokens = np.concatenate([seed, np.zeros(size - ls)])
    for i in range(ls, size):
        probs = model.predict([tokens[None,:], z])
        next_token = util.sample_logits(probs[0, i-1, :], temperature=temperature) # Extract the i-th probability vector and sample an index from it
        tokens[i] = next_token
    result = [int(t) for t in tokens]
    return result

def load_encoder_and_decoder():
    ## Define encoder
    input = Input(shape=(None, ), name='inp')
    embedding = Embedding(numwords, options.embedding_size, input_length=None)
    embedded = embedding(input)
    encoder = LSTM(options.lstm_capacity) if options.rnn_type == 'lstm' else GRU(options.lstm_capacity)
    h = Bidirectional(encoder)(embedded)
    tozmean = Dense(options.hidden)
    zmean = tozmean(h)
    tozlsigma = Dense(options.hidden)
    zlsigma = tozlsigma(h)
    ## Define KL Loss and sampling
    kl = util.KLLayer(weight = K.variable(1.0)) # computes the KL loss and stores it for later
    zmean, zlsigma = kl([zmean, zlsigma])
    eps = Input(shape=(options.hidden,), name='inp-epsilon')
    sample = Sample()
    zsample = sample([zmean, zlsigma, eps])
    ## Define decoder
    input_shifted = Input(shape=(None, ), name='inp-shifted')
    expandz_h = Dense(options.lstm_capacity, input_shape=(options.hidden,))
    z_exp_h = expandz_h(zsample)
    expandz_c = Dense(options.lstm_capacity, input_shape=(options.hidden,))
    z_exp_c = expandz_c(zsample)
    state = [z_exp_h, z_exp_c]
    seq = embedding(input_shifted)
    seq = SpatialDropout1D(rate=options.dropout)(seq)
    decoder_rnn = LSTM(options.lstm_capacity, return_sequences=True)
    h = decoder_rnn(seq, initial_state=state)
    towords = TimeDistributed(Dense(numwords))
    out = towords(h)
    auto = Model([input, input_shifted, eps], out)
    ## Extract the encoder and decoder models form the autoencoder
    encoder = Model(input, [zmean, zlsigma])
    encoder.load_weights("./py/lstm_sentence_level_chatbot/encoder.h5")
    z_in = Input(shape=(options.hidden,))
    s_in = Input(shape=(None,))
    seq = embedding(s_in)
    z_exp_h = expandz_h(z_in)
    z_exp_c = expandz_c(z_in)
    state = [z_exp_h, z_exp_c]
    h = decoder_rnn(seq, initial_state=state)
    out = towords(h)
    decoder = Model([s_in, z_in], out)
    decoder.load_weights("./py/lstm_sentence_level_chatbot/decoder.h5")
    return (encoder, decoder)

class Sample(Layer):
    """
    Performs sampling step
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var, eps = inputs

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _, _ = input_shape
        return shape_mu
