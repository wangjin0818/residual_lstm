# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence
from keras.layers import Input
from keras.optimizers import RMSprop, Adagrad, Adadelta
from keras.layers.merge import concatenate

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from utils import keras_pearsonr


# maxlen = 56
batch_size = 10
nb_epoch = 10
hidden_dim = 120
lstm_layer = 1

kernel_size = 3
nb_filter = 120
maxlen = 40

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x

def make_idx_data(revs, word_idx_map, maxlen=maxlen):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['intensity']

        if rev['option'] == 'train':
            X_train.append(sent)
            y_train.append(y)
        elif rev['option'] == 'test':
            X_test.append(sent)
            y_test.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    y_train= np.array(y_train)
    y_test = np.array(y_test)

    return [X_train, X_test, y_train, y_test]

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'cvat_wiki.pickle3')
    revs, W, word_idx_map, vocab, maxlen  = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')
    # maxlen=40

    X_train, X_test, y_train, y_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("max features of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension num of word vector [num_features]: %d" % num_features)

    # Keras Model
    sequence = Input(shape=(maxlen, ), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)

    if lstm_layer == 1:
        forwards = LSTM(hidden_dim) (embedded)
        forwards = Dropout(0.25) (forwards)

        backwards = LSTM(hidden_dim, go_backwards=True) (embedded)
        backwards = Dropout(0.25) (backwards)
    else:
        forwards = LSTM(hidden_dim, return_sequences=True) (embedded)
        forwards = Dropout(0.25) (forwards)

        backwards = LSTM(hidden_dim, go_backwards=True, return_sequences=True) (embedded)
        backwards = Dropout(0.25) (backwards)

        for i in range(2, lstm_layer):
            forwards = LSTM(hidden_dim, return_sequences=True) (forwards)
            forwards = Dropout(0.25) (forwards)

            backwards = LSTM(hidden_dim, go_backwards=True, return_sequences=True) (backwards)
            backwards = Dropout(0.25) (backwards)

        forwards = LSTM(hidden_dim) (forwards)
        forwards = Dropout(0.25) (forwards)

        backwards = LSTM(hidden_dim, go_backwards=True) (backwards)
        backwards = Dropout(0.25) (backwards)

    merged = concatenate([forwards, backwards], axis=-1)
    lstm = Dropout(0.25) (merged)

    output = Dense(1, activation='linear') (lstm)

    model = Model(inputs=sequence, outputs=output)

    optimizer = Adadelta(lr=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae', keras_pearsonr])

    model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=batch_size, epochs=nb_epoch, verbose=2)
    y_pred = model.predict(X_test, batch_size=batch_size).flatten()
    # print(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # r = pearsonr(y_test, y_pred)[0]
    r = pearsonr(y_test, y_pred)[0]
    logging.info('mean squared error: %.3f, mean absolute error: %.3f, pearsonr: %.3f' % (mse, mae, r))
