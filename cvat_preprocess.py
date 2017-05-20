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

import jieba
jieba.set_dictionary(os.path.join('dict', 'dict.txt.big'))

# configuration
num_feature = 400

# random state 3
random_state = 3
test_size = 0.2

def rescale(value):
    return (value - 1.) / 8.

def load_2100_corpus():
    corpus_file = os.path.join('corpus', 'cvat', 'new_2100_corpus.csv')
    corpus = pd.read_csv(corpus_file, header=0, delimiter=',', quoting=3)

    data, valence, arousal = [], [], []
    for i in range(len(corpus['sentence'])):
        text = corpus['sentence'][i]
        valence.append(round(corpus['avl_valence'][i], 2))
        arousal.append(round(corpus['avl_arouse'][i], 2))
        # print(text + ' ' + str(valence) + ' ' + str(arousal))

        words = jieba.cut(text, cut_all=False)
        curline = []
        for word in words:
            curline.append(word)
        # print(curline)
        data.append(curline)
    return data, valence, arousal

# def build_data_train_test(data, valence, arousal):
#     """
#     loads data and split into train and test sets.
#     """
#     revs = []
#     vocab = defaultdict(float)

#     train_data, test_data, train_valence, test_valence, train_arousal, test_arousal = \
#         train_test_split(data, valence, arousal, test_size=test_size, random_state= random_state)

#     for i in xrange(len(train_data)):
#         if np.isnan(train_valence[i]) or np.isnan(train_arousal[i]):
#             print(i)

#         if not np.isnan(train_valence[i]) and not np.isnan(train_arousal[i]):
#             line = train_data[i]

#             rev = []
#             orig_rev = ' '.join(line)
#             words = set(orig_rev.split())
#             for word in words:
#                 vocab[word] += 1
#             datum = {
#                 'valence': train_valence[i],
#                 'arousal': train_arousal[i],
#                 'text': orig_rev,
#                 'num_words': len(orig_rev.split()),
#                 'split': 'train'
#             }
#             revs.append(datum)

#     for i in xrange(len(test_data)):
#         if not np.isnan(test_valence[i]) and not np.isnan(test_arousal[i]):
#             line = test_data[i]

#             rev = []
#             orig_rev = ' '.join(line)
#             words = set(orig_rev.split())
#             for word in words:
#                 vocab[word] += 1
#             datum = {
#                 'valence': test_valence[i],
#                 'arousal': test_arousal[i],
#                 'text': orig_rev,
#                 'num_words': len(orig_rev.split()),
#                 'split': 'test'
#             }
#             revs.append(datum)

#     return revs, vocab

# def load_bin_vec(model, vocab):
#     word_vecs = {}
#     unk_words = 0

#     for word in vocab.keys():
#         try:
#             word_vec = model[word]
#             word_vecs[word] = word_vec
#         except:
#             unk_words = unk_words + 1
    
#     logging.info('unk words: %d' % (unk_words))
#     return word_vecs

# def get_W(word_vecs, k=400):
#     vocab_size = len(word_vecs)
#     word_idx_map = dict()

#     W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
#     W[0] = np.zeros((embedding_dim, ))
#     W[1] = np.random.uniform(-0.25, 0.25, k)

#     i = 2
#     for word in word_vecs:
#         W[i] = word_vecs[word]
#         word_idx_map[word] = i
#         i = i + 1
#     return W, word_idx_map

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    data, valence, arousal = load_2100_corpus()
    with codecs.open(os.path.join('corpus', 'cvat', 'new_2100_cvat.txt'), 'w', 'utf8') as my_file:
        for i in range(len(data)):
            my_file.write('%s,%f,%f\n' % (' '.join(data[i]), valence[i], arousal[i]))

    # revs, vocab = build_data_train_test(data, valence, arousal)
    # max_l = np.max(pd.DataFrame(revs)['num_words'])
    # mean_l = np.mean(pd.DataFrame(revs)['num_words'])
    # std_l = np.std(pd.DataFrame(revs)['num_words'])
    # logging.info('data loaded!')
    # logging.info('number of sentences: ' + str(len(revs)))
    # logging.info('vocab size: ' + str(len(vocab)))
    # logging.info('max sentence length: ' + str(max_l))
    # logging.info('mean sentence length: ' + str(mean_l))
    # logging.info('std sentence length: ' + str(std_l))

    # model_file = os.path.join('vector', 'wiki.zh.fan.model')
    # model = gensim.models.KeyedVectors.load(model_file)

    # w2v = load_bin_vec(model, vocab)
    # logging.info('word2vec loaded!')
    # logging.info('num words in word2vec: ' + str(len(w2v)))

    # W, word_idx_map = get_W(w2v, k=model.vector_size)
    # logging.info('extracted index from word2vec! ')



    # pickle_file = os.path.join('pickle', 'cvat_wiki.pickle3')
    # pickle.dump([revs, W, word_idx_map, vocab], open(pickle_file, 'wb'))
    # logging.info('dataset created!')

