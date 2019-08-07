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

"""AddOneNgram class."""

from pagi.utils.ngram import NGram


class AddOneNGram(NGram):
  """Add one n-Gram."""

  def __init__(self, n, sents, corpus=''):
    NGram.__init__(self, n, sents, corpus='')
    # way more efficient than using set union
    voc = ['</s>']
    for s in sents:
      voc += s
    self.voc = list(set(voc))
    self.corpus = corpus
    self.smoothingtechnique = 'Add One (Laplace) Smoothing'
    sents = list(map((lambda x: x + ['</s>']), sents))

  def cond_prob(self, token, prev_tokens=None):
    """Conditional probability of a token.
    token -- the token.
    prev_tokens -- the previous n-1 tokens (optional only if n = 1).
    """
    if not prev_tokens:
      assert self.n == 1
      prev_tokens = tuple()

    prev_tuple = tuple(prev_tokens)
    hits = self.count((prev_tuple+(token,)))
    sub_count = self.count(prev_tuple)

    # hits = self.count((tuple(prev_tokens)+(token,)))
    # sub_count = self.count(tuple(prev_tokens))

    del prev_tuple

    # heuristic
    return (hits+1) / (float(sub_count)+self.V())

  def V(self):
    """Size of the vocabulary.
    """
    return len(self.voc)
