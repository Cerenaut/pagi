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

from pagi.utils.embedding import Embedding

class SemanticEmbedding(Embedding):
  """
  Embedding transforms a large number of disjoint classes (e.g. words) into a dense or sparse combination of fewer dimensions.
  Good embeddings have other features represented in the relationships between classes.
  """

  def create(self, corpus_files, model_file, shape=[10,10], sparsity=0, eos='<end>'):
    size = np.prod(shape[:])
    print('----> Size: ', size)

    self.clear()

    data = self.read_corpus_files(corpus_files, eos)

    num_lines = len(data)
    logging.info('Corpus has %d lines.', num_lines)
    window_size = 5 #10 #2 
    min_count = 1
    workers = 10
    epochs = 20
    #sg = 0 # CBOW
    sg = 1 # skip-gram
    model = gensim.models.Word2Vec(data, size=size, window=window_size, min_count=min_count, workers=10, iter=epochs, sg=sg)    

    #model.train(data, total_examples=num_lines, epochs=10)


    words = list(model.wv.vocab)
    num_words = len(words)
    logging.info('Model has %d words.', num_words)

    word_indices = list(range(num_words))

    #print(" WORDS len: ", len(words))
    model.wv.save_word2vec_format(model_file, binary=False)
    logging.info('Wrote model to file: ' + model_file)

    w1 = ['asbestos']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['cigarettes']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['cents']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['million']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['elaborate']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['massachusetts']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['company']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)

    w1 = ['england']
    similar = model.wv.most_similar(positive=w1, topn=6)
    print( 'Similar: ', w1, ' and ', similar)



    if False:
      from sklearn.decomposition import PCA
      from matplotlib import pyplot
      X = model[model.wv.vocab]
      pca = PCA(n_components=2)
      result = pca.fit_transform(X)
      # create a scatter plot of the projection
      pyplot.scatter(result[:, 0], result[:, 1])
      words = list(model.wv.vocab)
      for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
      pyplot.show()

  def set_op(self, name, op, shape=None, default_value=None):
    if name in self._duals.keys():
      dual = self._duals[name]
    else:
      dual = self.add(name, shape, default_value)
    dual.set_op(op)

  def clear(self):
    self._matrix = None
    self._keyIndex = None # key: word, value: index
    self._indexKey = None

  def has_keys(self, keys):
    num_keys = len(keys)
    for i in range(num_keys):
      key = keys[i]
      if key not in self._keyIndex:
        logging.error('Key "' + key + '" not found.')
        return False

    return True

  def get_num_keys(self):
    return len(self._indexKey)

  def get_num_values(self):
    return self._matrix.shape[1]

  def get_index(self, key):
    index = self._keyIndex[key]
    return index

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

  # def read_csv(self, file_path):
  #   self.clear()

  #   with open(file_path, 'rt') as csvfile:
  #     sample = csvfile.read(1024)
  #     csvfile.seek(0)
  #     sniffer = csv.Sniffer()
  #     has_header = sniffer.has_header(sample)
  #     logging.info( 'Has header? ' + str(has_header))
  #     dialect = sniffer.sniff(sample)
  #     #csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
  #     csvreader = csv.reader(csvfile, dialect)

  #     num_rows = 0
  #     num_cols = None

  #     for row in csvreader:
  #       row_cols = len(row) -1
  #       if num_cols is not None:
  #         if row_cols != num_cols:
  #           logging.error('Row has inconsistent number of cols.')
  #           return False

  #       num_cols = row_cols
  #       num_rows += 1

  #     if has_header:
  #       num_rows -= 1
  
  #     #logging.info('Found ' + str(num_rows) + ' rows.')        

  #     self._matrix = np.zeros([num_rows, num_cols])
  #     self._keyIndex = {}
  #     self._indexKey = []#np.zeros(num_rows)

  #     csvfile.seek(0)
  #     row_index = 0
  #     first_row = True
  #     for row in csvreader:
  #       if first_row:
  #         first_row = False
  #         if has_header:
  #           continue

  #       key = row[0]

  #       # stupid parser cant deal
  #       if key == '</s>':
  #         key = '/s'
  #       if key == '<unk>':
  #         key = 'unk'

  #       for c in range(num_cols):
  #         str_val = row[c+1]
  #         #print( 'row: ', row_index, " key: ", key, " val: ", str_val, ' c', c)
  #         value = float(str_val)
  #         self._matrix[row_index][c] = value

  #       self._keyIndex.update({key:row_index})
  #       self._indexKey.append(key)#[row_index] = key

  #       #logging.info('Key: ' + str(key) + ' index: ' + str(row_index))

  #       row_index += 1

  #   return True


if __name__ == '__main__':
  util.set_logging('debug')
  e = GensimEmbedding()
  e.create(['/home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt'], 'model.txt' )
  # x = e.read_corpus_files(['/home/dave/agi/penn-treebank/simple-examples/data/ptb.train.txt'])
   
  # # build vocabulary and train model
  # model = gensim.models.Word2Vec(
  #       x,
  #       size=150,
  #       window=10,
  #       min_count=2,
  #       workers=10)
  # model.train(x, total_examples=len(x), epochs=10)
 
