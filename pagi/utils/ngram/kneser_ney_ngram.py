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

"""KneserNeyNGram class."""

from pagi.utils.ngram.kne import AddOneNGram


# pylint: disable-all
class KneserNeyNGram(KneserNeyBaseNGram):
  """
  Kneser-Ney smoothing.

  From https://west.uni-koblenz.de/sites/default/files/BachelorArbeit_MartinKoerner.pdf
  """

  def __init__(self, sents, words, n, corpus='', estimate_D=False, discount=None):
    super(KneserNeyNGram, self).__init__(sents=sents, words=words, n=n, corpus=corpus,
                                         estimate_D=estimate_D, discount=discount)

  def cond_prob(self, token, prev_tokens=None):
    n = self.n
    # two cases:
    # 1) n == 1
    # 2) n > 1:
        # 2.1) k == 1
        # 2.2) 1 < k < n
        # 2.3) k == n

    # case 1)
    # heuristic addone
    if not prev_tokens and n == 1:
      return (self.count((token,))+1) / (self.count(()) + self.V())

    # case 2.1)
    # lowest ngram
    if not prev_tokens and n > 1:
      aux1 = len(self.N_dot_tokens((token,)))
      aux2 = self.N_dot_dot()
      # addone smoothing
      return (aux1 + 1) / (aux2 + self.V())

    # highest ngram
    if len(prev_tokens) == n-1:
      c = self.count(prev_tokens) + 1
      t1 = max(self.count(prev_tokens+(token,)) - self.D, 0) / c
      # addone smoothing
      t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / c
      t3 = self.cond_prob(token, prev_tokens[1:])
      return t1 + t2 * t3
    # lower ngram
    else:
      # addone smoothing
      aux = max(len(self.N_dot_tokens_dot(prev_tokens)), 1)
      t1 = max(len(self.N_dot_tokens(prev_tokens+(token,))) - self.D, 0) / aux
      t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / aux
      t3 = self.cond_prob(token, prev_tokens[1:])
      return t1 + t2 * t3

  def cond_prob_fast(self, token, prev_tokens=None):
    #print('fast')
    n = self.n
    # two cases:
    # 1) n == 1
    # 2) n > 1:
        # 2.1) k == 1
        # 2.2) 1 < k < n
        # 2.3) k == n

    # case 1)
    # heuristic addone
    if not prev_tokens and n == 1:
      #print('1')
      tt = (token,)
      et = ()
      p = (self.count_get(tt)+1) / (self.count_get(et) + self.V())
      del tt
      del et
      return p

    # case 2.1)
    # lowest ngram
    if not prev_tokens and n > 1:
      #print('2')
      tt = (token,)
      aux1 = len(self.N_dot_tokens_get(tt))
      aux2 = self.N_dot_dot()
      # addone smoothing
      p = (aux1 + 1) / (aux2 + self.V())
      del tt
      return p

    # highest ngram
    if len(prev_tokens) == n-1:
      tt = (token,)
      pt = prev_tokens+tt
      xt = prev_tokens[1:]

      c = self.count_get(prev_tokens) + 1
      t1 = max(self.count_get(pt) - self.D, 0) / c
      # addone smoothing
      t2 = self.D * max(len(self.N_tokens_dot_get(prev_tokens)), 1) / c
      t3 = self.cond_prob_fast(token, xt)
      p = t1 + t2 * t3

      del tt
      del pt
      del xt
      return p
    # lower ngram
    else:
      tt = (token,)
      pt = prev_tokens+tt
      xt = prev_tokens[1:]

      # addone smoothing
      aux = max(len(self.N_dot_tokens_dot_get(prev_tokens)), 1)
      #t1 = max(len(self.N_dot_tokens(prev_tokens+(token,))) - self.D, 0) / aux
      t1 = max(len(self.N_dot_tokens_get(pt)) - self.D, 0) / aux
      t2 = self.D * max(len(self.N_tokens_dot_get(prev_tokens)), 1) / aux
      t3 = self.cond_prob_fast(token, xt)
      p = t1 + t2 * t3

      del tt
      del pt
      del xt
      return p

  def mod_cond_prob_fast(self, token, prev_tokens=None):
    #print('fast')
    n = self.n
    # two cases:
    # 1) n == 1
    # 2) n > 1:
        # 2.1) k == 1
        # 2.2) 1 < k < n
        # 2.3) k == n

    # case 1)
    # heuristic addone
    if not prev_tokens and n == 1:
      #print('1')
      tt = (token,)
      et = ()
      p = (self.count_get(tt)+1) / (self.count_get(et) + self.V())
      return p

    # case 2.1)
    # lowest ngram
    if not prev_tokens and n > 1:
      #print('2')
      tt = (token,)
      aux1 = len(self.N_dot_tokens_get(tt))
      aux2 = self.N_dot_dot()
      # addone smoothing
      p = (aux1 + 1) / (aux2 + self.V())
      return p

    # highest ngram
    if len(prev_tokens) == n-1:
      tt = (token,)
      pt = prev_tokens+tt
      xt = prev_tokens[1:]

      c = self.count_get(prev_tokens) + 1
      t1 = max(self.count_get(pt) - self.D, 0) / c
      # addone smoothing
      t2 = self.D * max(len(self.N_tokens_dot_get(prev_tokens)), 1) / c
      t3 = self.cond_prob_fast(token, xt)
      p = t1 + t2 * t3
      return p
    # lower ngram
    else:
      tt = (token,)
      pt = prev_tokens+tt
      xt = prev_tokens[1:]

      # addone smoothing
      aux = max(len(self.N_dot_tokens_dot_get(prev_tokens)), 1)
      #t1 = max(len(self.N_dot_tokens(prev_tokens+(token,))) - self.D, 0) / aux
      t1 = max(len(self.N_dot_tokens_get(pt)) - self.D, 0) / aux
      t2 = self.D * max(len(self.N_tokens_dot_get(prev_tokens)), 1) / aux
      t3 = self.cond_prob_fast(token, xt)
      p = t1 + t2 * t3
      return p
