# Copyright (C) 2019 Project AGI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Embedding"""

import logging
import numpy as np
import tensorflow as tf
import csv
import gensim

class Embedding(object):
  """
  Embedding transforms a large number of disjoint classes (e.g. words) into a dense or sparse combination of fewer dimensions.
  Good embeddings have other features represented in the relationships between classes.

  self._matrix = np.zeros([num_rows, num_cols]) The embedding
  self._keyIndex = {} given key, what's the index (reverse lookup)
  self._indexKey = [] array where index has relevant key

  """

  def __init__(self):
    self.clear()

  @staticmethod
  def tokenize_sentence(sentence, delimiter=' '):
    words = sentence.strip().split(delimiter)
    clean_words = []
    for word in words:
      clean_word = word.strip()
      if len(clean_word) > 0:
        clean_words.append(clean_word)
    return clean_words

  def read_corpus_files(self, corpus_files, eos='<end>'):
    #print("files: ", corpus_files)
    #import nltk.data

    text = ''
    for i in range(len(corpus_files)):
      file = corpus_files[i]
      print( "reading file", file)
      corpus = open(file).read()
      #print( "corp ", corpus)
      text = text + corpus + '\n'

    #sent_tokenize_list = nltk.sent_tokenize(all_text)
    sentences = text.splitlines()
    #tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    
    #sentences = tokenizer.tokenize(all_text)
    #print( "Here3 sentences", sentences)
    #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    #sentenized = doc_set.body.apply(sent_detector.tokenize)
    #sentences = itertools.chain.from_iterable(sentenized.tolist()) # just to flatten

    data = []
    for sentence in sentences:
      # #result += [nltk.word_tokenize(sent)]
      # words = sentence.strip().split(' ')
      # clean_words = []
      # for word in words:
      #   clean_word = word.strip()
      #   if len(clean_word) > 0:
      #     clean_words.append(clean_word)
      clean_words = Embedding.tokenize_sentence(sentence)
      if len(clean_words) > 0:
        clean_words.append(eos) # add EOS marker, which is needed to embed this token
        data.append(clean_words)
    #print( "result: ", data)
    return data

  def create(self, corpus_files, model_file, size=100, eos='<end>'):

    self.clear()

    # data = self.read_corpus_files(corpus_files, eos)

    # # https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    # model = gensim.models.Word2Vec(data, size=size, min_count=1)    
    # words = list(model.wv.vocab)
    # #print(" WORDS len: ", len(words))
    # model.wv.save_word2vec_format(model_file, binary=False)
    # logging.info('Wrote model to file: ' + model_file)

    # # from sklearn.decomposition import PCA
    # # from matplotlib import pyplot
    # # X = model[model.wv.vocab]
    # # pca = PCA(n_components=2)
    # # result = pca.fit_transform(X)
    # # # create a scatter plot of the projection
    # # pyplot.scatter(result[:, 0], result[:, 1])
    # # words = list(model.wv.vocab)
    # # for i, word in enumerate(words):
    # #   pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # # pyplot.show()

  def set_op(self, name, op, shape=None, default_value=None):
    if name in self._duals.keys():
      dual = self._duals[name]
    else:
      dual = self.add(name, shape, default_value)
    dual.set_op(op)

  def check(self):
    num_keys = len(self._indexKey)
    num_cols = len(self._matrix[0])

    min_overlap = num_cols * num_cols
    max_overlap = 0
    #sum_overlap = 0
    #max_bits = 0

    for i in range(num_keys):
      print('i=',i, 'max overlap ', max_overlap)
      for j in range(num_keys):
        if i == j:
          continue

        if i > j:
          continue

        overlap = 0
        bits = 0

        for k in range(num_cols):
          x_i = self._matrix[i][k]
          x_j = self._matrix[j][k]
          if (x_i > 0.0):
            if (x_j > 0.0):
              overlap = overlap +1
            bits = bits +1

        #print('i=',i,' j=',j, 'max overlap ', max_overlap, 'overlap ', overlap, ' max bits: ', max_bits)

        #sum_overlap += overlap
        min_overlap = min(min_overlap, overlap)
        max_overlap = max(max_overlap, overlap)
        #max_bits = max(max_bits, bits)

    #mean_overlap = float(sum_overlap) / float(max_bits)

    print('Overlap min: ', min_overlap, ' of ', max_bits)
    print('Overlap max: ', max_overlap, ' of ', max_bits)
    #print('Overlap avg: ', mean_overlap, ' of ', max_bits)

  def clear(self):
    self._matrix = None
    self._keyIndex = None # key: word, value: index
    self._indexKey = None # key: index, value: word

  def has_keys(self, keys):
    num_keys = len(keys)
    for i in range(num_keys):
      key = keys[i]
      has_key = self.has_key(key)
      if not has_key:
        return False
    return True

  def has_key(self, key):
    if key not in self._keyIndex:
      logging.debug('Key "' + key + '" not found.')
      return False
    return True

  def get_keys(self):
    # keys = []
    # num_keys = len(self._indexKey)
    # for i in range(num_keys):
    #   keys.append(self._indexKey[i])
    return self._indexKey

  def get_num_keys(self):
    return len(self._indexKey)

  def get_num_values(self):
    return self._matrix.shape[1]

  def get_index(self, key):
    index = self._keyIndex[key]
    return index

  def get_key(self, index):
    key = self._indexKey[index]
    return key

  def get_value(self, key, index):
    row = self.get_index(key)
    value = self._matrix[row][index]
    return value

  def get_values(self, key):
    row = self.get_index(key)
    values = self._matrix[row]
    return values

  def read(self, file_path):
    self.clear()

    with open(file_path, 'rt') as file:

      rows = file.readlines()
      num_file_rows = len(rows)
      #print( "found ", num_file_rows , " rows ")

      row = rows[0]
      cols = row.split(' ')

      num_rows = int(cols[0])
      num_cols = int(cols[1])

      #print( "embedding has ", num_rows , " x ", num_cols)

      self._matrix = np.zeros([num_rows, num_cols])
      self._keyIndex = {}
      self._indexKey = []#np.zeros(num_rows)

      for r in range(num_rows):
        row = rows[r+1]
        #print("row: ", row)
        values = row.split(' ')
        key = values[0].strip()

        for c in range(num_cols):
          pass

          str_val = values[c+1]
          #print( 'row: ', row_index, " key: ", key, " val: ", str_val, ' c', c)
          value = float(str_val)
          self._matrix[r][c] = value

        self._keyIndex.update({key:r})
        self._indexKey.append(key)#[row_index] = key

        #logging.info('Key: ' + str(key) + ' index: ' + str(row_index))

    return True
