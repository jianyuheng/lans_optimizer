# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

import numpy as np
import random

class LANSOptimizer(tf.train.Optimizer):
    """
  LANSOptimizer optimizer.
  # References
  - Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes. http://arxiv.org/abs/2006.13484
  - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
  """

    def __init__(
        self,
        learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=None,
        name="LANSOptimizer",
    ):
        """Constructs a LAMBOptimizer."""
        super(LANSOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            grad /= tf.norm(grad, ord=2)

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/lans_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer(),
            )
            v = tf.get_variable(
                name=param_name + "/lans_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer(),
            )

            # Standard Adam update.
            next_m = tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + tf.multiply(
                1.0 - self.beta_2, tf.square(grad)
            )

            # apply bias correction
            steps = tf.cast(global_step + 1, tf.float32)
            beta1_correction = (1 - self.beta_1 ** steps)
            beta2_correction = (1 - self.beta_2 ** steps)
            next_m_unbiased = next_m / beta1_correction
            next_v_unbiased = next_v / beta2_correction
            update = next_m_unbiased / (tf.sqrt(next_v_unbiased) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############
            w_norm = tf.norm(param, ord=2)
            g_norm = tf.norm(update, ord=2)

            # calculate lans_trust_ratio for first part
            ratio_m = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

            first_part = self.learning_rate * ratio_m * self.beta_1 * update
            next_param = param - first_part

            # calculate the second part of the estimator
            update = grad / (tf.sqrt(next_v_unbiased) + self.epsilon)

            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * next_param

            g_norm = tf.norm(update, ord=2)
            ratio_g = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

            second_part = self.learning_rate * ratio_g * (1 - self.beta_1) * update

            next_param = next_param - second_part

            assignments.extend(
                [param.assign(next_param), m.assign(next_m), v.assign(next_v)]
            )
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
