import keras

import keras.backend as K
from keras.datasets import imdb
from keras.layers import \
    Layer, Dense, LSTM, Embedding, TimeDistributed, Bidirectional, SpatialDropout1D, GRU, Input
from keras.models import Model
from tensorflow.python.client import device_lib

from tensorboardX import SummaryWriter

from keras.utils import multi_gpu_model

from tqdm import tqdm
import math, sys, os, random
import numpy as np
import util

CHECK = 5
NINTER = 10
HIDDEN_LAYER_SIZE = 16
LSTM_CAPACITY = 256
EMBEDDING_SIZE = 300
encoder, auto, decoder = load_encoder_and_decoder()


def response(user_input, diversity=1.2, response_length=100):
    response_text = ""
    x, w21, i2w = util.format_words(user_input)
    x = util.batch_pad(x, add_eos=True)
    b = random.choice(x)
    z, _ = encoder.predict(b)
    z = z[None, 0, :]
    gen = generate_seq(decoder, z=z, size=response_length, temperature=diversity)
    # Show the argmax reconstruction
    n, _ = b.shape
    b_shifted = np.concatenate([np.ones((n, 1)), b], axis=1)  # prepend start symbol
    eps = np.random.randn(n, HIDDEN_LAYER_SIZE)   # random noise for the sampling layer
    out = auto.predict([b, b_shifted, eps])[None, 0, :]
    out = np.argmax(out[0, ...], axis=1)
    util.decode([int(o) for o in out])
    for i in range(CHECK):
        # Sample two z's from N(0,1), interpolate between them; greedy decoding: i.e. pick word with highest probability
        zfrom, zto = np.random.randn(1, HIDDEN_LAYER_SIZE), np.random.randn(1, HIDDEN_LAYER_SIZE)
        for d in np.linspace(0, 1, num=NINTER):
            z = zfrom * (1-d) + zto * d
            gen = generate_seq(decoder, z=z, size=response_length, temperature=0.0)
            response += " " + util.decode(gen)
    return response

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
    embedding = Embedding(util.VOCAB_SIZE, EMBEDDING_SIZE, input_length=None)
    embedded = embedding(input)
    encoder = LSTM(LSTM_CAPACITY)
    h = Bidirectional(encoder)(embedded)
    tozmean = Dense(HIDDEN_LAYER_SIZE)
    zmean = tozmean(h)
    tozlsigma = Dense(HIDDEN_LAYER_SIZE)
    zlsigma = tozlsigma(h)
    ## Define KL Loss and sampling
    kl = KLLayer(weight = K.variable(1.0)) # computes the KL loss and stores it for later
    zmean, zlsigma = kl([zmean, zlsigma])
    eps = Input(shape=(HIDDEN_LAYER_SIZE,), name='inp-epsilon')
    sample = Sample()
    zsample = sample([zmean, zlsigma, eps])
    ## Define decoder
    input_shifted = Input(shape=(None, ), name='inp-shifted')
    expandz_h = Dense(LSTM_CAPACITY, input_shape=(HIDDEN_LAYER_SIZE,))
    z_exp_h = expandz_h(zsample)
    expandz_c = Dense(LSTM_CAPACITY, input_shape=(HIDDEN_LAYER_SIZE,))
    z_exp_c = expandz_c(zsample)
    state = [z_exp_h, z_exp_c]
    seq = embedding(input_shifted)
    seq = SpatialDropout1D(rate=0.5)(seq)
    decoder_rnn = LSTM(LSTM_CAPACITY, return_sequences=True)
    h = decoder_rnn(seq, initial_state=state)
    towords = TimeDistributed(Dense(numwords))
    out = towords(h)
    auto = Model([input, input_shifted, eps], out)
    ## Extract the encoder and decoder models form the autoencoder
    encoder = Model(input, [zmean, zlsigma])
    encoder.load_weights("./py/lstm_sentence_level_chatbot/encoder.h5")
    z_in = Input(shape=(HIDDEN_LAYER_SIZE,))
    s_in = Input(shape=(None,))
    seq = embedding(s_in)
    z_exp_h = expandz_h(z_in)
    z_exp_c = expandz_c(z_in)
    state = [z_exp_h, z_exp_c]
    h = decoder_rnn(seq, initial_state=state)
    out = towords(h)
    decoder = Model([s_in, z_in], out)
    decoder.load_weights("./py/lstm_sentence_level_chatbot/decoder.h5")
    return (encoder, auto, decoder)

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


class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.

    During training, call
            K.set_value(kl_layer.weight, new_value)
    to scale the KL loss term.

    based on:
    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight=None, *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        loss = K.mean(kl_batch)
        if self.weight is not None:
            loss = loss * self.weight

        self.add_loss(loss, inputs=inputs)

        return inputs
