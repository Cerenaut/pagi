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

"""DenseEmbedding"""

import logging
import numpy as np
import tensorflow as tf
import csv
import gensim

from pagi.utils.embedding import Embedding

class DenseEmbedding(Embedding):
  """
  Produces a dense and highly overlapping embedding for each token.
  """

  def get_num_bits(self, num_tokens):
    num_bits = 1
    while (1<<num_bits) < num_tokens:
      print('num_tokens: ', num_tokens, ' num_bits: ', num_bits, ' (1<<num_bits): ', (1<<num_bits) )
      num_bits += 1
    print('num_tokens: ', num_tokens, ' num_bits: ', num_bits, ' (1<<num_bits): ', (1<<num_bits) )
    assert((1<<num_bits) > num_tokens)
    return num_bits

  def get_unique_tokens(self, corpus_files, eos):
    data = self.read_corpus_files(corpus_files, eos)

    tokens = set()

    for sentence in data:
      for word in sentence:
        tokens.add(word)

    # convert the set to the list 
    unique_tokens = (list(tokens)) 
    return unique_tokens

  def create_matrix(self, unique_tokens, num_bits):
    """Create a 2-d matrix of the embeddings. The first dim is words or tokens. The 2nd is the embedding vector"""    
    num_tokens = len(unique_tokens)
    num_draws = 1  # TODO allow multiple random shuffles
    height = num_tokens
    width = num_bits * num_draws * 2
    #depth = 1

    matrix = np.zeros([height, width])

    # Each row is a different token (word)
    for row in range(num_tokens):
      token = unique_tokens[row]
      np.zeros(width)

      # convert the row id into a bit pattern
      binary_str = str('{0:0'+str(num_bits)+'b}').format(row)
      print('Row: ', row, ' bits: ', binary_str)

      # Double the bits by including not(bits) to give constant density
      # e.g. 
      # Row:  9502  
      # bits:  [1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 
      #         0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1.]

      # Each 
      for bit in range(num_bits):
        value = int(binary_str[bit])
        matrix[row][bit] = value
        matrix[row][bit + num_bits] = 1 - value
        # matrix[row][bit][0 = value
        # matrix[row][bit][1] = 1 - value

      print('Row: ', row, ' bits: ', matrix[row])

    return matrix

  def create_shape(self, corpus_files, eos):
    unique_tokens = self.get_unique_tokens(corpus_files, eos)
    num_tokens = len(unique_tokens)
    num_bits = self.get_num_bits(num_tokens)
    embedding_shape = [num_bits, 2]
    return embedding_shape

  def create(self, corpus_files, model_file, shape=[10,10], sparsity=20, eos='<end>'):

    self.clear()

    unique_tokens = self.get_unique_tokens(corpus_files, eos)

    # Now generate a random matrix for each 
    num_tokens = len(unique_tokens)
    #print( "found ", num_tokens, " tokens" )

    num_bits = self.get_num_bits(num_tokens)

    matrix = self.create_matrix(unique_tokens, num_bits)
    self.write(model_file, matrix, unique_tokens)
    logging.info('Wrote model to file: ' + model_file)

    embedding_shape = self.get_embedding_shape(num_bits)  #[num_bits, 2]
    return embedding_shape

  def write(self, file_path, matrix, tokens):

    num_tokens = len(tokens)

    with open(file_path, mode='w') as file:

      content = ''  
      for row in range(num_tokens):
        token = tokens[row]
        values = matrix[row]    
        num_cols = len(values)

        if row == 0:
          content += (str(num_tokens) + ' ' +str(num_cols))

        row_values = token + ' '
        for col in range(num_cols):
          value = values[col]
          row_values += (str(value) + ' ')

        content += ('\n' + row_values) 

      file.write(content)

