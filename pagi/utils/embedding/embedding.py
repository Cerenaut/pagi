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

"""Embedding class."""

import sys
import logging

import numpy as np


class Embedding:
  """
  A numerical representation of a set of tokens as an n-d array.
  In practice we assume the token values are a 2-d matrix per token.

  self._token_values = [token, values_h, values_w] The embedding.
  self._token_index = {} given token, what's the index (reverse lookup)
  self._index_token = [] array where index has relevant token
  """

  def __init__(self):
    """Noarg constructor and deferred setup()"""
    self.clear()

  def clear(self):
    self._delimiter = ' '
    self._token_values = None
    self._token_index = None # key: word, value: index
    self._index_token = None # key: index, value: word

  @staticmethod
  def tokenize_line(sentence, delimiter=' '):
    words = sentence.strip().split(delimiter)
    clean_words = []
    for word in words:
      clean_word = word.strip()
      if clean_word:
        clean_words.append(clean_word)
    return clean_words

  @staticmethod
  def tokenize_files(corpus_files, delimiter=' ', eos='<end>'):
    """Read the text files of a corpus into an array of tokens."""
    text = ''
    for file in corpus_files:
      corpus = open(file).read()
      text = text + corpus + '\n'  # Add a trailing newline to concat files

    sentences = text.splitlines()

    data = []
    for sentence in sentences:
      tokens = Embedding.tokenize_line(sentence, delimiter)
      if tokens:
        if eos is not None:
          tokens.append(eos) # add EOS marker, which is needed to embed this token
        data.extend(tokens)
    # print("result: ", data)
    return data

  def has_tokens(self, tokens):
    num_tokens = len(tokens)
    for i in range(num_tokens):
      token = tokens[i]
      has_token = self.has_token(token)
      if not has_token:
        return False
    return True

  def has_token(self, token):
    if token not in self._token_index.keys():
      logging.debug('Token "%s" not found.', token)
      return False
    return True

  def get_tokens(self):
    """Returns an array of tokens"""
    return self._index_token

  def get_num_tokens(self):
    """Returns number of unique tokens"""
    return len(self._index_token)

  def get_token_value_area(self):
    value_shape = self.get_token_value_shape()
    area = np.prod(value_shape)
    return area

  def get_token_value_shape(self):
    """Returns the shape of a single token's values"""
    return self._token_values.shape[1:]

  def get_index(self, token):
    index = self._token_index[token]
    return index

  def get_token(self, index):
    token = self._index_token[index]
    return token

  def get_tokens_values(self):
    return self._token_values

  def get_token_value(self, token, y, x):
    index = self.get_index(token)
    value = self._token_values[index][y][x]
    return value

  def get_token_values(self, token):
    index = self.get_index(token)
    values = self._token_values[index]
    return values

  def check(self):
    """Check that the shapes match and tokens have unique vectors."""
    token_values_shape = self.get_token_value_shape()
    num_trees = token_values_shape[0]
    num_values = token_values_shape[1]
    num_tokens = len(self._index_token)

    min_sum_sq_diff = sys.float_info.max

    for index1 in range(num_tokens):
      token1 = self._index_token[index1]
      print('check token:', index1, 'which is', token1)      
      #min_sum_sq_diff = sys.float_info.max  # reset 

      for index2 in range(num_tokens):

        if index2 == index1:
          continue

        token2 = self._index_token[index2]
        print('compare token:', index2, 'which is', token2)

        for t in range(num_trees):
          sum_sq_diff = 0.0
          for v in range(num_values):
            x1 = self._token_values[index1][t][v]
            x2 = self._token_values[index2][t][v]
            sum_sq_diff = sum_sq_diff + ((x1-x2) * (x1-x2))
          #print('tree diff = ', sum_sq_diff)      
          min_sum_sq_diff = min(min_sum_sq_diff, sum_sq_diff)

      print('check token:', index1, 'which is', token1, ' min diff = ', min_sum_sq_diff)      
    print('All tokens min diff = ', min_sum_sq_diff)      
    threshold = 1.0
    if min_sum_sq_diff >= threshold:
      return True
    return False

  def write_tokens_values(self, file_path):
    np.save(file_path, self._token_values)

  def read_tokens_values(self, file_path):
    self._token_values = np.load(file_path)

  def write_tokens(self, file_path, delimiter=' '):
    np.savetxt(file_path, self._index_token, delimiter=delimiter, fmt='%s')

  def read_tokens(self, file_path, delimiter=' '):
    """Read token excluding embeddings."""
    #self._index_token = np.genfromtxt(file_path, dtype='str')
    self._index_token = Embedding.tokenize_files([file_path], delimiter, eos=None)
    #print('index token: ', self._index_token)

    self._token_index = {}
    num_tokens = len(self._index_token)
    for index in range(num_tokens):
      token = self._index_token[index]
      self._token_index.update({token: index})
