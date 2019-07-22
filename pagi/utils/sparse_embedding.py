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

"""SparseEmbedding"""

import logging
import numpy as np
import tensorflow as tf
import csv
import gensim

from pagi.utils.embedding import Embedding

class SparseEmbedding(Embedding):
  """
  Produces a sparse and highly orthogonal embedding for each token.
  """

  def create(self, corpus_files, model_file, shape=[10,10], sparsity=20, eos='<end>'):

    self.clear()

    data = self.read_corpus_files(corpus_files, eos)

    tokens = set()

    for sentence in data:
      for word in sentence:
        tokens.add(word)

    # convert the set to the list 
    unique_tokens = (list(tokens)) 

    # Now generate a random matrix for each 
    num_tokens = len(unique_tokens)
    #print( "found ", num_tokens, " tokens" )
    num_rows = num_tokens
    num_cols = np.prod(shape[:])

    matrix = np.zeros([num_rows, num_cols])
    for row in range(num_tokens):
      token = unique_tokens[row]
      np.zeros(num_cols)
      for bit in range(sparsity):
        col = np.random.randint(num_cols)
        matrix[row][col] = 1.0

    self.write(model_file, matrix, unique_tokens)
    logging.info('Wrote model to file: ' + model_file)

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

