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

"""Kneser Ney Smoothing"""

# https://docs.python.org/3/library/collections.html
from math import log
from collections import defaultdict


# Due to memory leak, don't want to add all the test samples to the dict.
# https://stackoverflow.com/questions/49778527/suppress-key-addition-in-collections-defaultdict
class DefaultDict(defaultdict):
  def get_and_forget(self, key):
    return self.get(key, self.default_factory())  # pylint: disable=not-callable


class NGram:
  """
  Downloaded from:
  https://github.com/giovannirescia/PLN-2015
  Specifically:
  https://github.com/giovannirescia/PLN-2015/blob/practico4/languagemodeling/ngram.py
  """

  def __init__(self, n, sents, corpus='', sos='<s>', eos='</s>'):
    """
    n -- order of the model.
    sents -- list of sentences, each one being a list of tokens.
    corpus -- which corpus is being used
    """
    assert n > 0
    self.n = n
    #self.counts = counts = defaultdict(int)
    self.counts = counts = DefaultDict(int)
    self.corpus = corpus

    # Prefix and suffix sentences with SOS and EOS tokens
    sents = list(map((lambda x: [sos]*(n-1) + x), sents))
    sents = list(map((lambda x: x + [eos]), sents))

    for sent in sents:
      #print('sent ', sent, ' of ', len(sents))
      for i in range(len(sent) - n + 1):
        ngram = tuple(sent[i: i + n])
        counts[ngram] += 1
        counts[ngram[:-1]] += 1

  # obsolete now...
  def prob(self, token, prev_tokens=None):
    n = self.n
    if not prev_tokens:
      prev_tokens = []
    assert len(prev_tokens) == n - 1

    tokens = prev_tokens + [token]
    aux_count = self.counts[tuple(tokens)]
    return aux_count / float(self.counts[tuple(prev_tokens)])

  def count(self, tokens):
    """Count for an n-gram or (n-1)-gram.
    tokens -- the n-gram or (n-1)-gram tuple.
    """
    #print('size: ', len(self.counts))
    return self.counts[tokens]

  def cond_prob(self, token, prev_tokens=None):
    """Conditional probability of a token.
    token -- the token.
    prev_tokens -- the previous n-1 tokens (optional only if n = 1).
    """

    if not prev_tokens:
      assert self.n == 1
      prev_tokens = tuple()
    # ngram condicional probs are based on relative counts
    hits = self.count((tuple(prev_tokens)+(token,)))
    sub_count = self.count(tuple(prev_tokens))

    return hits / float(sub_count)

  def sent_prob(self, sent):
    """Probability of a sentence. Warning: subject to underflow problems.
    sent -- the sentence as a list of tokens.
    """

    prob = 1.0
    sent = ['<s>']*(self.n-1)+sent+['</s>']

    for i in range(self.n-1, len(sent)):
      prob *= self.cond_prob(sent[i], tuple(sent[i-self.n+1:i]))
      if not prob:
        break

    return prob

  def sent_log_prob(self, sent):
    """Log-probability of a sentence.
    sent -- the sentence as a list of tokens.
    """

    prob = 0
    sent = ['<s>']*(self.n-1)+sent+['</s>']

    for i in range(self.n-1, len(sent)):
      c_p = self.cond_prob_fast(sent[i], tuple(sent[i-self.n+1:i]))
      # to catch a math error
      if not c_p:
        return float('-inf')
      prob += log(c_p, 2)

    return prob

  def perplexity(self, sents):
    """ Perplexity of a model.
    sents -- the test corpus as a list of sents
    """
    # total words seen
    m = 0
    for sent in sents:
      m += len(sent)
    # cross-entropy
    l = 0
    print('Computing Perplexity on {} sents...\n'.format(len(sents)))
    for sent in sents:
      l += self.sent_log_prob(sent) / m
    return pow(2, -l)

  def get_special_param(self):
    return None, None
