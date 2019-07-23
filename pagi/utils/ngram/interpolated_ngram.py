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

"""InterpolatedNGram class."""

from pagi.utils.ngram import AddOneNGram


# pylint: disable-all
class InterpolatedNGram(AddOneNGram):

  def __init__(self, n, sents, corpus='', gamma=None, addone=True):
    """
    n -- order of the model.
    sents -- list of sentences, each one being a list of tokens.
    gamma -- interpolation hyper-parameter (if not given, estimate using
        held-out data).
    addone -- whether to use addone smoothing (default: True).
    corpus -- which corpus is being used
    """
    self.n = n
    self.smoothingtechnique = 'Interpolated (Jelinek Mercer) Smoothing'
    self.gamma = gamma
    self.addone = addone
    #self.counts = counts = defaultdict(int)
    self.counts = counts = DefaultDict(int)
    self.gamma_flag = True
    self.corpus = corpus
    # way more efficient than use set unions
    voc = ['</s>']
    for s in sents:
      voc += s
    self.voc = list(set(voc))

    if gamma is None:
      self.gamma_flag = False

    # if not gamma given
    if not self.gamma_flag:
      total_sents = len(sents)
      aux = int(total_sents * 90 / 100)
      # 90 per cent for training
      train_sents = sents[:aux]
      # 10 per cent for perplexity (held out data)
      held_out_sents = sents[-total_sents+aux:]

      train_sents = list(map((lambda x: ['<s>']*(n-1) + x), train_sents))
      train_sents = list(map((lambda x: x + ['</s>']), train_sents))

      for sent in train_sents:
        for j in range(n+1):
          # move along the sent saving all its j-grams
          for i in range(n-j, len(sent) - j + 1):
            ngram = tuple(sent[i: i + j])
            counts[ngram] += 1
      # added by hand
      counts[('</s>',)] = len(train_sents)
      # variable only for tests
      self.tocounts = counts
      # search the gamma that gives lower perplexity
      gamma_candidates = [i*50 for i in range(1, 15)]
      # xs is a list with (gamma, perplexity)
      xs = []
      sents = train_sents
      for aux_gamma in gamma_candidates:
        self.gamma = aux_gamma
        aux_perx = self.perplexity(held_out_sents)
        xs.append((aux_gamma, aux_perx))
      xs.sort(key=lambda x: x[1])
      self.gamma = xs[0][0]
      with open('old-stuff/interpolated_' + str(n) + '_parameters_'+corpus, 'a') as f:
        f.write('Order: {}\n'.format(self.n))
        f.write('Gamma: {}\n'.format(self.gamma))
        f.write('AddOne: {}\n'.format(self.addone))
        f.write('Perplexity observed: {}\n'.format(xs[0][1]))
        f.write('-------------------------------\n')
      f.close()

    else:
      sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))
      sents = list(map((lambda x: x + ['</s>']), sents))

      for sent in sents:
        # counts now holds all k-grams for 0 < k < n + 1
        for j in range(n+1):
          # move along the sent saving all its j-grams
          for i in range(n-j, len(sent) - j + 1):
            ngram = tuple(sent[i: i + j])
            counts[ngram] += 1
      # added by hand
      counts[('</s>',)] = len(sents)

  def cond_prob(self, token, prev_tokens=None):
    """Conditional probability of a token.
    token -- the token.
    prev_tokens -- the previous n-1 tokens (optional only if n = 1).
    """

    addone = self.addone
    n = self.n
    gamma = self.gamma

    if not prev_tokens:
      prev_tokens = []
      assert len(prev_tokens) == n - 1

    lambdas = []
    for i in range(0, n-1):
      # 1 - sum(previous lambdas)
      aux_lambda = 1 - sum(lambdas)
      # counts for numerator
      counts_top = self.count(tuple(prev_tokens[i:n-1]))
      # counts plus gamma (denominator)
      counts_w_gamma = self.count(tuple(prev_tokens[i:n-1])) + gamma
      # save the computed i-th lambda
      lambdas.append(aux_lambda * (counts_top / counts_w_gamma))
    # last lambda, by hand
    lambdas.append(1-sum(lambdas))

    # Maximum likelihood probs
    ML_probs = dict()
    for i in range(0, n):
      hits = self.count((tuple(prev_tokens[i:])+(token,)))
      sub_count = self.count(tuple(prev_tokens[i:]))
      result = 0
      if addone and not len(prev_tokens[i:]):
        result = (hits+1) / (float(sub_count) + len(self.voc))
      else:
        if sub_count:
          result = hits / float(sub_count)
      # the (i+1)-th element in ML_probs holds the q_ML value
      # for a (n-i)-gram
      ML_probs[i+1] = result

    prob = 0
    # ML_probs dict starts in 1
    for j in range(0, n):
      prob += ML_probs[j+1] * lambdas[j]
    return prob

  def get_special_param(self):
    return "Gamma", self.gamma
