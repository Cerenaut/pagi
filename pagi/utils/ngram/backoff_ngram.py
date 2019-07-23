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

"""BackOffNGram class."""

from pagi.utils.ngram import NGram


# pylint: disable-all
class BackOffNGram(NGram):

  def __init__(self, n, sents, corpus='', beta=None, addone=True):
    """
    Back-off NGram model with discounting as described by Michael Collins.
    n -- order of the model.
    sents -- list of sentences, each one being a list of tokens.
    beta -- discounting hyper-parameter (if not given, estimate using
        held-out data).
    addone -- whether to use addone smoothing (default: True).
    corpus -- which corpus is being used
    """
    self.n = n
    self.beta = beta
    self.corpus = corpus
    self.beta_flag = True
    self.addone = addone
    self.smoothingtechnique = 'Back Off (Katz) with Discounting Smoothing'
    #self.counts = counts = defaultdict(int)
    self.counts = counts = DefaultDict(int)
    self.A_set = defaultdict(set)
    voc = ['</s>']
    for s in sents:
      voc += s
    self.voc = set(voc)
    if beta is None:
      self.beta_flag = False

    # if no beta given, we compute it
    if not self.beta_flag:
      total_sents = len(sents)
      aux = int(total_sents * 90 / 100)
      # 90 per cent por training
      train_sents = sents[:aux]
      # 10 per cent for perplexity (held out data)
      held_out_sents = sents[-total_sents+aux:]

      train_sents = list(map((lambda x: ['<s>']*(n-1) + x), train_sents))
      train_sents = list(map((lambda x: x + ['</s>']), train_sents))
      for sent in train_sents:
        for j in range(n+1):
          for i in range(n-j, len(sent) - j + 1):
            ngram = tuple(sent[i: i + j])
            counts[ngram] += 1
            # for efficiency, we save the A set as a dict of sets
            if j:
              self.A_set[ngram[:-1]].add(ngram[-1])
      for i in range(1, n):
        counts[('<s>',)*i] += len(train_sents)
      counts[('</s>',)] = len(train_sents)

      self.tocounts = counts
      # search for the beta that gives lower perplexity
      beta_candidates = [i*0.1 for i in range(1, 10)]
      # xs is a list with (beta, perplexity)
      xs = []
      self.sents = train_sents
      for aux_beta in beta_candidates:
        self.beta = aux_beta
        aux_perx = self.perplexity(held_out_sents)
        xs.append((aux_beta, aux_perx))
      xs.sort(key=lambda x: x[1])
      self.beta = xs[0][0]
      with open('old-stuff/backoff_'+str(n)+'_parameters_'+corpus, 'a') as f:
        f.write('Order: {}\n'.format(self.n))
        f.write('Beta: {}\n'.format(self.beta))
        f.write('AddOne: {}\n'.format(self.addone))
        f.write('Perplexity observed: {}\n'.format(xs[0][1]))
        f.write('-------------------------------\n')
      f.close()
    else:
      sents = list(map((lambda x: x + ['</s>']), sents))
      sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))

      for sent in sents:
        for j in range(n+1):
          for i in range(n-j, len(sent) - j + 1):
            ngram = tuple(sent[i: i + j])
            counts[ngram] += 1
            # for efficiency, we save the A set as a dict of sets
            if j:
              self.A_set[ngram[:-1]].add(ngram[-1])
      for i in range(1, n):
        counts[('<s>',)*i] += len(sents)
      counts[('</s>',)] = len(sents)

  # c*() counts
  def count_star(self, tokens):
    """
    Discounting counts for counts > 0
    """
    return self.counts[tokens] - self.beta

  def A(self, tokens):
    """Set of words with counts > 0 for a k-gram with 0 < k < n.
    tokens -- the k-gram tuple.
    """

    if not tokens:
      tokens = []
    return self.A_set[tuple(tokens)]

  def alpha(self, tokens):
    """Missing probability mass for a k-gram with 0 < k < n.
    tokens -- the k-gram tuple.
    """
    if not tokens:
      tokens = tuple()

    A_set = self.A(tokens)
    result = 1
    # heuristic, way more efficient
    if len(A_set):
      result = self.beta * len(A_set) / self.count(tuple(tokens))
    return result

  def cond_prob(self, token, prev_tokens=None):
    """Conditional probability of a token.
    token -- the token.
    prev_tokens -- the previous n-1 tokens (optional only if n = 1).
    """

    addone = self.addone

    # unigram case
    if not prev_tokens:
      if addone:
        result = (self.count((token,))+1) / (self.V() + self.count(()))
      else:
        result = self.count((token,)) / self.count(())
    else:
      A_set = self.A(prev_tokens)
      # check if discounting can be applied
      if token in A_set:
        result = self.count_star(tuple(prev_tokens) + (token,)) /\
            self.count(tuple(prev_tokens))
      else:
        # recursive call
        q_D = self.cond_prob(token, prev_tokens[1:])
        denom_factor = self.denom(prev_tokens)
        if denom_factor:
          alpha = self.alpha(prev_tokens)
          result = alpha * q_D / denom_factor
        else:
          result = 0
    return result

  def denom(self, tokens):
    """Normalization factor for a k-gram with 0 < k < n.
    tokens -- the k-gram tuple.
    """

    sum = 0
    A_set = self.A(tokens)
    # heuristic
    for elem in A_set:
      sum += self.cond_prob(elem, tokens[1:])
    return 1 - sum

  def V(self):
    """Size of the vocabulary.
    """
    return len(self.voc)

  def get_special_param(self):
    return "Beta", self.beta
