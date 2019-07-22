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

"""GngComponent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
from os.path import dirname, abspath

import numpy as np
import tensorflow as tf

from pagi.utils import image_utils
from pagi.utils.dual import DualData
from pagi.utils.layer_utils import activation_fn
from pagi.utils.np_utils import np_write_filters
from pagi.classifier.component import Component


class GngComponent(Component):
  """
  Growing Neural Gas (a winner-take-all competitive learning algorithm).
  """

  @staticmethod
  def default_hparams():
    """Builds an HParam object with default hyperparameters."""
    return tf.contrib.training.HParams(
        batch_size=100,
        cells=100,
        edge_max_age=1000,
        learning_rate=0.01,
        learning_rate_neighbours=0.001,
        stress_learning_rate=0.01,
        stress_split_learning_rate=0.01,
        stress_threshold=0.0,
        utility_learning_rate=0.01,
        utility_threshold=0.0,
        growth_interval=0.0
    )

  def __init__(self):
    self._name = None
    self._hparams = None
    self._dual = None

  def build(self, input_values, input_shape, hparams, name='component'):
    """Initializes the model parameters.

    Args:
        input_values: Tensor containing input
        input_shape: The shape of the input, for display (internal is vectorized)
        hparams: The hyperparameters for the model as tf.contrib.training.HParams.
        name: A globally unique graph name used as a prefix for all tensors and ops.
    """
    self._name = name
    self._hparams = hparams
    self._dual = DualData(self._name)
    self._input_values = input_values
    self._input_shape = input_shape
    self._age_since_growth = 0

    with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
      self._build()

    self.reset()

  def reset(self):
    self._age_since_growth = 0
    self._dual.set_values('utility', 0.0)
    self._dual.set_values('stress', 0.0)

    # Randomize weights
    weights = np.random.rand(self._weights_shape)
    self._dual.set_values('weights', weights)

  @property
  def name(self):
    return self._name

  def get_dual(self):
    return self._dual

  def _build(self):
    """Build the graph ops"""

    input_shape = input_tensor.get_shape().as_list()
    input_area = np.prod(input_shape[1:])
    input_vector_shape = [self._hparams.batch_size, input_area]

    input_cells_shape = [self._hparams.cells, input_area]
    cells_shape = [self._hparams.cells]

    utility_pl = self._dual.add('utility', shape=cells_shape, default_value=0.0).add_pl()
    stress_pl = self._dual.add('stress', shape=cells_shape, default_value=0.0).add_pl()
    ages_pl = self._dual.add('ages', shape=cells_shape, default_value=0.0).add_pl()
    #self._dual.add('edge_ages', shape=cells_shape, default_value=0.0).add_pl()

    self._dual.add('weights', shape=input_cells_shape, default_value=0.0).add_pl()

    # update
    input_1d = tf.reshape(self._input_values, input_vector_shape)  # [b,i]

    # mean sq error
    weights = self._dual.get('weights')
    batch_weights = tf.expand_dims(weights, 0)  # now [1,c,i]
    batch_cells_inputs = tf.expand_dims(input_1d, 1)  # [b,1,i]
    sq_err = tf.squared_difference(batch_weights, batch_cells_inputs)
    cell_sq_err = tf.reduce_sum(sq_err, axis=2) # shape=[b,c] reduce per cell, per batch
    self._set_op('error', tf.reduce_sum(cell_sq_err))

    # find min, then exclude that, then find min again for neighbour.
    # finds the best cell per batch
    min_error_1 = tf.reduce_min(cell_sq_err, axis=[1], keepdims=True)  # [b,1]
    top1_mask_bool = tf.less_equal(cell_sq_err, min_error_1)  # [b,c]
    top1_mask = tf.to_float(top1_mask_bool)  # [b,c]
    self._dual.set_op('top1', top1_mask)

    # increase error of top1 to exclude it
    #not_1_mask = 1.0 - top_1_mask  # all 1s except the top-1
    max_mask = top_1_mask * sys.float_info.max
    ex_top1_sq_err = cell_sq_err + max_mask

    min_error_2 = tf.reduce_min(ex_top1_sq_err, axis=[1], keepdims=True)  # [b,1]
    top2_mask_bool = tf.less_equal(ex_top1_sq_err, min_error_2)
    top2_mask = tf.to_float(top2_mask_bool)
    self._dual.set_op('top2', top2_mask)

    # update edges
    #           = [b,c1,1 ]  A
    #           = [b,1, c1]  B
    #           = [b,c1,1 ]  A
    #           = [b,1, c1]  B
    # edge_mask = [b,c1,c2]
    # edges = [c1,c2]
    edge_mask_1a = tf.expand_dims(top1_mask,1)
    edge_mask_1b = tf.transpose(edge_mask_1a)
    edge_mask_2a = tf.expand_dims(top2_mask,1)
    edge_mask_2b = tf.transpose(edge_mask_2a)

    # edges have no direction, so there are 2 edges
    # There should be 2 bits per batch sample in these structures
    edge_mask_x = tf.multiply(edge_mask_1a, edge_mask_2b)
    edge_mask_y = tf.multiply(edge_mask_2a, edge_mask_1b)
    batch_edge_mask = tf.maximum(edge_mask_x, edge_mask_y)  # combine the 2 edges

    # reduce the edges over the batch
    edge_mask = tf.reduce_max(batch_edge_mask)  # Contains all the edges in the batch

    # set edge ages to zero if they were '1'
    # so the mask must be 
    edges_pl = self._dual.get_pl('edges')
    edges_ages_pl = self._dual.get_pl('edges_ages')

    # prune edges by age?
    edges_age_mask_bool = tf.less(edges_ages_pl, self._hparams.edge_max_age)
    edges_age_mask = tf.to_float(edges_age_mask_bool)
    edges_masked = edges_pl * edges_age_mask  # Only retain edges < max age

??????

    edges_aged = edges_pl + 1.0
    edges_aged = tf.minium(edges_aged, self._hparams.edge_max_age)
    inv_edge_mask = 1.0 - edge_mask
    edges_ages_op = edges_aged * inv_edge_mask  # ie new edges' ages = 0
    self._dual.set_op('edges_ages', edges_op)  # updated edges ages

    edges_op = tf.maximum(edges_aged, edge_mask)  # combine new and old edges
    self._dual.set_op('edges', edges_op)  # updated edges

    # update winning cells' ages
    top1_mask_1d = tf.reduce_max(top1_mask, axis=0)  # reduce over batch
    inv_top1_mask = 1.0 - top1_mask_1d
    aged = ages_pl + 1.0
    ages_op = aged * inv_top1_mask  # ie top1 cells' ages = 0
    self._dual.set_op('ages', ages_op)  # updated edges


    # OK reduceStress();
    # OK float cellStressLearningRate = _c.getStressLearningRate();
    # OK     _cellStress._values[ i ] -= _cellStress._values[ i ] * cellStressLearningRate;
    stress_decay = stress_pl - (stress_pl * self._hparams.stress_learning_rate)

    # updateStress();
    # float bestSumSqError = _cellErrors._values[ _bestCell ]; // use abs errors instead of sq errors?
    # float stressOld = _cellStress._values[ _bestCell ];
    # float stressNew = stressOld + bestSumSqError; // ( float ) Unit.lerp( stressOld, bestSumSqError, cellStressAlpha );
    # _cellStress._values[ _bestCell ] = stressNew;
    stress_mask = top1_mask * cell_sq_err
    stress_op = stress_decay + stress_mask
    self._dual.set_op('stress', stress_op)

    # // train winner A and its neighbours towards the input
    # trainCells();
    # protected void updateWeight( int cell, int inputs, int i, float inputValue, float cellLearningRate )
    #    float weightNew = weightOld + cellLearningRate * ( inputValue - weightOld );

    # Already:
    # batch_weights = tf.expand_dims(weights_pl, 0)  # now [1,c,i]
    # batch_cells_inputs = tf.expand_dims(input_1d, 1)  # [b,1,i]

    # top1_mask = [b,c]
    # n_mask = the neighbours (not the actual cell)
    # NOTE: Neighbours must NOT be self. Let's assume that.
    n_mask = all n for each c in the current b
    TODO

    mask_1_3d = tf.expand_dims(top1_mask, 2) # [b,c,1]
    mask_n_3d = tf.expand_dims(n_mask, 2) # [b,c,1] where in c, all n are '1's

    weight_error = tf.subtract(batch_cells_inputs, batch_weights)  # [b,c,i]
    batch_weight_change_1 = mask_1_3d * weight_error * self._hparams.learning_rate
    batch_weight_change_n = mask_n_3d * weight_error * self._hparams.learning_rate_neighbours
    weight_change_1 = tf.reduce_sum(batch_weight_change_1, axis=0)  # Assume samples cancel out if not same direction
    weight_change_n = tf.reduce_sum(batch_weight_change_n, axis=0)

    weights_op = weights_pl + weight_change_1 + weight_change_n
    self._dual.set_op('weights', weights_op)


    # OK reduceUtility();
    # OK _cellUtility._values[ i ] -= _cellUtility._values[ i ] * cellUtilityLearningRate;
    utility_decay = utility_pl - (utility_pl * self._hparams.utility_learning_rate)

    # OK updateUtility();
    #   // increase utility of winner
    #   // U_winner = U_winner + error_2nd - error_winner
    #   // So if error_winner = 0 and error_2nd = 1
    #   // utility = 1-0
    #   // if winner = 1 and error = 1.1, then
    #   // utility = 1.1 - 1 = 0.1 (low utility, because 2nd best almost as good)
    #   // error B >= A by definition, cos A won.
    #
    #   float sumSqErrorA = _cellErrors._values[ _bestCell ]; // use abs errors instead of sq errors?
    #   float sumSqErrorB = _cellErrors._values[ _2ndBestCell ]; // use abs errors instead of sq errors?
    #   float utility = sumSqErrorB - sumSqErrorA; // error B >= A by definition, cos A won.
    #   float utilityOld = _cellUtility._values[ _bestCell ];
    #   float utilityNew = utilityOld + utility;
    #   _cellUtility._values[ _bestCell ] = utilityNew;
    batch_utility = min_error_2 - min_error_1  # error B >= A by definition, cos A won. So pos.
    cell_utility = top1_mask * batch_utility  # Only add this utility to the winning cell
    utility_op = utility_decay + cell_utility

    # OK recycle cells...

    # OK updateCellsAges();


  def maintenance(self):
    # OFF GRAPH
    # // Create new cells where the space is poorly represented (the cells are stressed)
    # int ageSinceGrowth = ( int ) _ageSinceGrowth._values[ 0 ];
    # int growthInterval = _c.getGrowthInterval();
    # if( ageSinceGrowth >= growthInterval ) {
    #     removeLowUtilityCell();
    #     if( addCells() ) {
    #         ageSinceGrowth = 0;
    #     }
    # }
    #
    #_ageSinceGrowth._values[ 0 ] = ageSinceGrowth + 1;
    growth_interval = self._hparams.growth_interval
    if self._age_since_growth >= growth_interval:
      #removeLowUtilityCell()
      #addCells()
      self.recycle_cell()
      self._age_since_growth = 0

  def recycle_cell(self):
    # Picks a cell to recycle, and makes it the split of two others
    oldest_cell = self.find_oldest_cell()
    stressed_cell_1, stressed_cell_2 = self.find_stressed_cells()
    if stressed_cell_1 is None:
      return
    if stressed_cell_2 is None:
      return
    self.bisect_cells( stressed_cell_1, stressed_cell_2, oldest_cell )

  def find_oldest_cell(self):
    # Since we're not having dead cells (fixed pop.), 
    # the most useless cell has the greatest age
    ages = self._dual.get_values('ages')  # updated edges
    cell1 = np.argmax(ages)  # greatest age = most useless
    return cell1

  def are_neighbours(self, cell1, cell2):
    edges = self._dual.get_values('edges')  # updated edges

    edge_index = cell1 * self.hparams.cells + cell2      
    edge_value = edges[edge_index]
    are_neighbours = (edge_value > 0.0)
    return are_neighbours

  def find_stressed_cells(self):

    stress = self._dual.get_values('stress')  # updated edges
    cell1 = np.argmax(stress)  # i = most stressed

    # Check if cells are stress enough
    stress1 = stress[cell1]
    if stress1 < self.hparams.stress_threshold:
      return None, None

    # Find the neighbour of cell1 that has the worst stress
    cell2 = None    
    stress_max = 0.0
    for i in range(self.hparams.cells):
      is_neighbour = self.are_neighbours(cell1, i)
      if is_neighbour is False:
        continue

      stress_i = stress[i]
      if stress_i >= stress_max:
        cell2 = i
        stress_max = stress_i

    return cell1, cell2  # May return none

  def bisect_cells(self, source_cell_1, source_cell_2, bisection_cell):

    # now we have the two cells we want to divide by putting a new cell between them
    # set weights of the new cell as median of the two worst cells
    input_area = self.get_input_area()
    weights = self._dual.get_values('weights')

    for j in range(input_area):
      w1 = weights[source_cell_1 * input_area + j]
      w2 = weights[source_cell_2 * input_area + j]
      w3 = (w1 + w2) * 0.5
      weights[bisection_cell * input_area + j] = w3;

    ages[bisection_cell] = 0  # make it young again

    # Remove all its old edges
    for i in range(self.hparams.cells):

    # Set the edge ages to 0

        // Create edges: worst, free; worst2, free;
        // remove edges: worst, worst2
        int offsetWW2 = getEdgeOffset( worstCell , worstCell2 );
        int offsetWF  = getEdgeOffset( worstCell, freeCell );
        int offsetW2F = getEdgeOffset( worstCell2, freeCell );

    edges_ages[] = 0
    edges_ages[] = 0
    edges_ages[] = 0

    edges[] = 0    # Remove edge
    edges[] = 1.0  # Create edge
    edges[] = 1.0  # Create edge

    # reset stress of all cells.
    # Note, this means I mustn't add any new cells until the new stresses have had time to work out
    stress.set_values_to(0)  # give it time to accumulate
    # float stressWorst1 = _cellStress._values[ worstCell  ];
    # float stressWorst2 = _cellStress._values[ worstCell2 ];
    # float stressFreeNew = ( stressWorst1 + stressWorst2 ) * 0.5f;

    # float cellStressSplitLearningRate = _c.getStressSplitLearningRate();
    # float stressWorst1New = stressWorst1 - (stressWorst1 * cellStressSplitLearningRate );
    # float stressWorst2New = stressWorst2 - (stressWorst2 * cellStressSplitLearningRate );

    # stress = self._dual.get_values('stress')
    # stress._values[source_cell_1 ] = stress1
    # stress._values[source_cell_2 ] = stress2
    # stress._values[bisection_cell] = stress3

    # bisect utility of new cell
    # U_new = ( U_worst1 + U_worst2 ) / 2
    utility = self._dual.get_values('utility')
    utility1 = utility[source_cell_1];
    utility2 = utility[source_cell_2];
    utility_value = ( utility1 + utility2 ) * 0.5
    utility._values[source_cell_1 ] = utility_value
    utility._values[source_cell_2 ] = utility_value
    utility._values[bisection_cell] = utility_value

    ages = self._dual.get_values('ages')
    ages[source_cell_1 ] = 0
    ages[source_cell_2 ] = 0
    ages[bisection_cell] = 0

    # Fritzke does not define the initialisation of the utility variable for a new node.
    # However, in the DemoGNG v1.5 implementation [6] it is defined as the mean of
    # Uu and Uv
    # Note also that there is no mention of a decrease of the utilities of nodes
    # u and v corresponding to the error decrease in GNG after a new node has been
    # inserted. It stands to reason that the utilities of u and v should be decreased in the
    # same manner as the errors.

  def update_feed_dict(self, feed_dict, batch_type='training'):
    # Everything the same, just doesn't overwrite off-graph weights if not learning
    mask = self._dual.get('mask')
    mask_pl = mask.get_pl()
    mask_values = mask.get_values()

    feed_dict.update({
        mask_pl: mask_values,
    })

  def add_fetches(self, fetches, batch_type='training'):
    # Everything the same, just doesn't overwrite off-graph weights if not learning
    fetches[self._name] = {
        'loss': self._dual.get_op('loss'),
        'training': self._dual.get_op('training'),
        'encoding': self._dual.get_op('encoding'),
        'decoding': self._dual.get_op('decoding')
    }

    if self._summary_training_op is not None:
      fetches[self._name]['summaries'] = self._summary_training_op

  def set_fetches(self, fetched, batch_type='training'):

    if batch_type == 'training':
      names = ['encoding', 'decoding']
    if batch_type == 'encoding':
      names = ['encoding', 'decoding']

    self._dual.set_fetches(fetched, names)

    if self._summary_training_op is not None:
      self._summary_values = fetched[self._name]['summaries']


  def build_summaries(self, batch_types=None, scope=None):
    """Builds all summaries."""
    if not scope:
      scope = self._name + '/summaries/'
    with tf.name_scope(scope):
      for batch_type in batch_types:
        if batch_type == 'training':
          self.build_training_summaries()
        if batch_type == 'encoding':
          self.build_encoding_summaries()
        if self._hparams.secondary and batch_type == 'secondary_encoding':
          pass
        if self._hparams.secondary and batch_type == 'secondary_decoding':
          self.build_secondary_decoding_summaries()

  def write_summaries(self, step, writer, batch_type='training'):
    """Write the summaries fetched into _summary_values"""
    if self._summary_values is not None:
      writer.add_summary(self._summary_values, step)
      writer.flush()


  def build_training_summaries(self):
    with tf.name_scope('training'):
      summaries = self._build_summaries()
      self._summary_training_op = tf.summary.merge(summaries)
      return self._summary_training_op

  def build_encoding_summaries(self):
    with tf.name_scope('encoding'):
      summaries = self._build_summaries()
      self._summary_encoding_op = tf.summary.merge(summaries)
      return self._summary_encoding_op

  def _build_summaries(self):
    """Build the summaries for TensorBoard."""
    max_outputs = 3
    summaries = []

    encoding_op = self.get_encoding_op()
    decoding_op = self.get_decoding_op()

    summary_input_shape = image_utils.get_image_summary_shape(self._input_shape)

    input_summary_reshape = tf.reshape(self._input_values, summary_input_shape)
    decoding_summary_reshape = tf.reshape(decoding_op, summary_input_shape)

    summary_reconstruction = tf.concat([input_summary_reshape, decoding_summary_reshape], axis=1)
    reconstruction_summary_op = tf.summary.image('reconstruction', summary_reconstruction,
                                                 max_outputs=max_outputs)
    summaries.append(reconstruction_summary_op)

    # show input on it's own
    input_alone = True
    if input_alone:
      summaries.append(tf.summary.image('input', input_summary_reshape, max_outputs=max_outputs))

    summaries.append(self._summary_hidden(encoding_op, 'encoding', max_outputs))

    # Loss
    loss_summary = tf.summary.scalar('loss', self._dual.get_op('loss'))
    summaries.append(loss_summary)

    # histogram of weights and hidden values
    summaries.append(tf.summary.histogram('weights', self._weights))
    summaries.append(tf.summary.histogram('encoding', self._dual.get_op('encoding')))

    # input_stats_summary = tf_build_stats_summaries(self._input_values, 'input-stats')
    # summaries.append(input_stats_summary)

    return summaries
