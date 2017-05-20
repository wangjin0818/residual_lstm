from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import gensim
import pickle
import codecs

import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.cross_validation import train_test_split

random_state=0

def rescale(value):
    return (value - 1.) / 8.

def build_data_train_test(x_train, x_test, y_train_valence, y_test_valence):
    revs = []
    vocab = defaultdict(float)

    for i in range(len(x_train)):
        if not np.isnan(y_train_valence[i]):
            line = x_train[i]
            orig_rev = ' '.join(line)
            words = set(orig_rev.split())

            for word in words:
                vocab[word] += 1
            datum = {
                'intensity': round(rescale(y_train_valence[i]), 2),
                'text': orig_rev,
                'num_words': len(orig_rev.split()),
                'option': 'train'
            }
            revs.append(datum)
    
    for i in range(len(x_test)):
        if not np.isnan(y_test_valence[i]):
            line = x_test[i]

            rev = []
            orig_rev = ' '.join(line)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                'intensity': round(rescale(y_test_valence[i]), 2),
                'text': orig_rev,
                'num_words': len(orig_rev.split()),
                'option': 'test'
            }
            revs.append(datum)

    return revs, vocab

# def load_bin_vec(model, vocab, k=400):
#     word_vecs = {}
#     unk_words = 0

#     for word in vocab.keys():
#         try:
#             word_vec = model[word]
#         except:
#             word_vec = np.random.uniform(-0.25, 0.25, k)
#             unk_words = unk_words + 1

#         word_vecs[word] = word_vec
    
#     logging.info('unk words: %d' % (unk_words))
#     return word_vecs

# def get_W(word_vecs, k=400):
#     vocab_size = len(word_vecs)
#     word_idx_map = dict()

#     W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
#     W[0] = np.zeros((k, ))
#     # W[1] = np.random.uniform(-0.25, 0.25, k)

#     i = 1
#     for word in word_vecs:
#         W[i] = word_vecs[word]
#         word_idx_map[word] = i
#         i = i + 1
#     return W, word_idx_map

def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1
    
    logging.info('unk words: %d' % (unk_words))
    return word_vecs

def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
    W[0] = np.zeros((k, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    cvat_file = os.path.join('corpus', 'cvat', 'new_2100_cvat.txt')
    cvat_df = pd.read_table(cvat_file, sep=',', header=None, quoting=3)

    text, intensity = [], []
    for i in range(len(cvat_df[0])):
        text.append(cvat_df[0][i])
        intensity.append(cvat_df[1][i])

    x_train, x_test, y_train, y_test = train_test_split(text, intensity, \
        random_state=random_state, test_size=0.2)

    revs, vocab = build_data_train_test(x_train, x_test, y_train, y_test)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    mean_l = np.mean(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))
    logging.info('mean sentence length: ' + str(mean_l))

    model_file = os.path.join('vector', 'wiki.zh.fan.model')
    model = gensim.models.KeyedVectors.load(model_file)
    print(dir(model))

    w2v = load_bin_vec(model, vocab)
    logging.info('word2vec loaded!')
    logging.info('num words in word2vec: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=400)
    logging.info('extracted index from word2vec! ')
    logging.info(W.shape)

    pickle_file = os.path.join('pickle', 'cvat_wiki.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')