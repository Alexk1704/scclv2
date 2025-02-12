#
# Copyright (c) 2021 Benedikt Bagus.
#
# This file is part of Approach to Solving the Catastrophic Forgetting Problem of Deep Learning Methods.
# See https://gitlab.cs.hs-fulda.de/ML-Projects for further info.
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
#
import tensorflow       as tf
import numpy            as np
import quadprog         as qp

from collections        import defaultdict
from tensorflow         import keras
from cl_replay.utils     import log
from keras.models       import Model
from keras.metrics      import Metric

class GEM(Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.buffer_size    = kwargs['buffer_size']
    self.balance_type   = kwargs['balance_type']
    self.selection_type = kwargs['selection_type'] # FIXME: not used!
    self.drop_model     = kwargs['drop_model']

    self.epsilon     = self.parser.add_argument('--epsilon'        , type=float, default=1e-3       , help='the learning rate')
    self.input_size  = self.parser.add_argument('--input_size'     , type=int  , default=[28, 28, 1], help='the input dimensions')
    self.num_classes = self.parser.add_argument('--num_classes'    , type=int  , default=10         , help='the output dimensions')

    self.averaged = self.parser.add_argument('--averaged', type=str  , default='no', choices=['no', 'yes'], help='should the gradients be averaged? [default="no", "yes"]')
    self.strength = self.parser.add_argument('--strength', type=float, default=0.5                        , help='the strength of the memories')

    self.current_task = 0

    # evaluation metrics
    metrics      = ['accuracy_score'] # + ['loss']
    self.metrics = self.parser.add_argument('--metrics', type=str, default=metrics, help='the evaluations metrics')
    self.metric  = Metric(self)

    # ---

    self.memories = []
    self.theta_shape = [component.shape for component in self.classifier.trainable_variables]

    self.classifier = self.build_classifier()
    self.loss_classifier = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.optimizer_classifier = tf.keras.optimizers.Adam(self.epsilon)
    # self.optimizer_classifier = tf.keras.optimizers.SGD(self.epsilon, momentum=0.9)

    #self.print_summary()

  def build_classifier(self):
    return tf.keras.Sequential([
      tf.keras.Input(self.input_size),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(400),
      tf.keras.layers.ReLU(),

      tf.keras.layers.Dense(self.num_classes, activation='softmax'),
    ], name='classifier')

  def print_summary(self):
    self.classifier.summary()

  def reset_model(self, task):
    self.current_task = task
    self.classifier = self.build_classifier()

  def build(self, **kwargs):
    log.info('build GEM model')
    return self

  def get_model_variables(self):
    return [self.classifier.trainable_variables]

  def gradient2theta(self, gradient):
    return tf.concat([tf.reshape(component, (-1,)) for component in gradient], axis=0)

  def theta2gradient(self, theta):
    index = 0
    gradient = []

    for component_shape in self.theta_shape:
      next_index = tf.add(index, tf.reduce_prod(component_shape))
      gradient.append(tf.reshape(theta[index:next_index], component_shape))
      index = next_index

    return gradient

  def project2orthogonal(self, gradient, memories):
    dot_product = tf.reduce_sum(tf.multiply(gradient, memories))
    memory_mag = tf.reduce_sum(tf.multiply(memories, memories))
    projection = tf.subtract(gradient, tf.multiply(tf.divide(dot_product, memory_mag), memories))

    return projection

  def project2cone(self, gradient, memories, eps=1e-3):
    gradient = np.array(gradient, dtype=np.float64)
    memories = np.array(memories, dtype=np.float64)

    t = memories.shape[0]
    P = np.dot(memories, memories.T)
    P = np.add(np.multiply(0.5, np.add(P, P.T)), np.multiply(np.eye(t), eps))
    q = np.negative(np.dot(memories, gradient))
    G = np.eye(t)
    h = np.add(np.zeros(t), self.strength)

    v = qp.solve_qp(P, q, G, h)[0]
    x = np.add(np.dot(v, memories), gradient)

    return np.array(x, dtype=np.float32)

  def handle_memories(self, data, labels):
    if len(self.memories) == self.current_task:
      self.memories.append({'data': None, 'labels': None})

      if self.current_task > 0:
        samples_per_task = self.buffer_size // len(self.memories)
        for task_index in range(len(self.memories) - 1):
          data = self.memories[task_index]['data'][-samples_per_task:]
          labels = self.memories[task_index]['labels'][-samples_per_task:]

          self.memories[task_index] = {'data': data, 'labels': labels}
    else:
      data = np.concatenate([self.memories[self.current_task]['data'], data], axis=0)
      labels = np.concatenate([self.memories[self.current_task]['labels'], labels], axis=0)

    samples_per_task = self.buffer_size // len(self.memories)
    if len(labels) > samples_per_task:
      data = data[-samples_per_task:]
      labels = labels[-samples_per_task:]

    self.memories[self.current_task] = {'data': data, 'labels': labels}

  def calculate_gradients(self, data, labels):
    with tf.GradientTape(persistent=True) as tape:
      gradients = []
      for previous_task in range(self.current_task):
        logits = self.classifier(self.memories[previous_task]['data'])
        loss = self.loss_classifier(self.memories[previous_task]['labels'], logits)
        gradient = tape.gradient(loss, self.classifier.trainable_variables)

        gradients.append(self.gradient2theta(gradient))

      logits = self.classifier(data)
      loss = self.loss_classifier(labels, logits)
      gradient = tape.gradient(loss, self.classifier.trainable_variables)

      gradients.append(self.gradient2theta(gradient))

    return gradient, gradients

  def validate_gradients(self, gradient, gradients):
    if self.current_task > 0:
      gradient_flatted = gradients[-1]
      memories_flatted = tf.stack(gradients[:-1])

      if self.averaged == 'yes':
        mean_memories_flatted = tf.reduce_mean(memories_flatted, axis=0)
        dot_product = tf.reduce_sum(tf.multiply(gradient_flatted, mean_memories_flatted))
        if dot_product < 0:
          gradient = self.theta2gradient(self.project2orthogonal(gradient_flatted, mean_memories_flatted))
      else:
        dot_product = tf.tensordot(tf.expand_dims(gradient_flatted, axis=0), tf.transpose(memories_flatted), axes=1)
        if tf.reduce_min(dot_product) < 0:
          gradient = self.theta2gradient(self.project2cone(gradient_flatted, memories_flatted))

    return gradient

  def train_step(self, xs, ys, **kwargs):
    self.handle_memories(xs, ys)

    gradient, gradients = self.calculate_gradients(xs, ys)
    gradient = self.validate_gradients(gradient, gradients)
    self.optimizer_classifier.apply_gradients(zip(gradient, self.classifier.trainable_variables))

  def test_step(self, xs, ys=None, **kwargs):
    logits   = self.classifier(xs)
    loss     = self.loss_classifier(ys, logits)

    y_pred   = tf.argmax(logits, axis=1)
    y_true   = tf.argmax(ys, axis=1)

    metric_values = self.metric.eval(
      dict    = True  ,
      loss    = loss  ,
      y_true  = y_true,
      y_pred  = y_pred,
      special = {
        'accuracy_score': dict(normalize=True),
        }
    )

    return {'GEM': metric_values}

  def test(self, test_iterator, **kwargs):
    results = defaultdict(list)
    for xs, ys in test_iterator:
      test_results = self.test_step(xs, ys)

      for layer_name, metric_and_values in test_results.items():
        for metric_name, metric_value in metric_and_values.items():
          results[(layer_name, metric_name)] += [metric_value]

    return_results = dict()
    for (layer_name, metric_name), metric_values in results.items():
      if 'accuracy' in metric_name: format_str  = '{:10.1%}'
      else                        : format_str  = '{:10.2f}'
      metric_values                             = np.mean(metric_values)
      return_results[(layer_name, metric_name)] = (metric_values, format_str.format(metric_values))

    return return_results
