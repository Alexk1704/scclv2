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

from collections        import defaultdict
from cl_replay.utils     import log
from cl_replay.parsing   import Kwarg_Parser

class NSR:
  def __init__(self, **kwargs):
    #super().__init__(**kwargs)

    self.parser = Kwarg_Parser(**kwargs) ;

    self.buffer_size    = self.parser.add_argument('--buffer_size'    , type=int, default=100       , help='the BS')

    self.replay_ratio   = self.parser.add_argument('--replay_ratio'    , type=float, default=1.       , help='the rr')

    self.balance_type   = self.parser.add_argument('--balance_type'    , type=str, default="tasks"       , help='the learning rate')

    self.partition_type = self.parser.add_argument('--partition_type'    , type=str, default=""       , help='the learning rate')

    self.selection_type = self.parser.add_argument('--selection_type'    , type=str, default=""       , help='the learning rate')
    self.drop_model     = self.parser.add_argument('--drop_model'    , type=str, default=""       , help='the learning rate')

    self.epsilon     = self.parser.add_argument('--epsilon'    , type=float, default=1e-3       , help='the learning rate')
    self.iterations  = self.parser.add_argument('--iterations' , type=int  , default=1          , help='number of iterations') # FIXME: not used!
    self.input_size  = self.parser.add_argument('--input_size' , type=int  , default=[28, 28, 1], help='the input dimensions')
    self.num_classes = self.parser.add_argument('--num_classes', type=int  , default=10         , help='the output dimensions')

    self.current_task = 0

    # evaluation metrics
    self.metric  = tf.keras.metrics.CategoricalAccuracy() ;


    # ---

    self.buffer = []
    self.classes = []

    if 'deviation' in self.selection_type:
      self.class_means = []
      self.mean_counter = 0

    if self.selection_type == 'reservoir':
      self.buffer = {'data': [], 'labels': []}
      self.reservoir_counter = 0

    self.classifier = self.build_classifier()
    self.loss_classifier = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.optimizer_classifier = tf.keras.optimizers.Adam(self.epsilon)
    # self.optimizer_classifier = tf.keras.optimizers.SGD(self.epsilon, momentum=0.9)

    # self.print_summary()

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

      tf.keras.layers.Dense(self.num_classes),
      tf.keras.layers.Softmax()
    ], name='classifier')

  def print_summary(self):
    self.classifier.summary()

  def reset_model(self, task):
    self.current_task = task
    self.classifier = self.build_classifier()

  def build(self, **kwargs):
    log.info('build NSR model')
    self.classifier.compile(loss=tf.keras.CategoricalCrossEntropy(), metrics = tf.keras.metrics.CategoricalAccuracy(), optimizer="Adam") ;
    return self

  def fit(self, *args, **kwargs):
    return self.classifier.fit(*args, **kwargs) ;


  def get_model_variables(self):
    return [self.classifier.trainable_variables]

  def reservoir_selection(self, data, labels):
    assert self.partition_type == 'none'

    for i in range(self.batch_size):
      if len(self.buffer['labels']) < self.buffer_size:
        self.buffer['data'].append(data[i])
        self.buffer['labels'].append(labels[i])
      else:
        random_index = np.random.randint(self.reservoir_counter)
        if random_index < self.buffer_size:
          self.buffer['data'][random_index] = data[i]
          self.buffer['labels'][random_index] = labels[i]
      self.reservoir_counter += 1

  def intensity_selection(self, index):
    assert self.partition_type == 'classes'

    intensities = np.sum(self.buffer[index]['data'], axis=tuple(range(1, self.buffer[index]['data'].ndim)))
    sorted_intensities = np.argsort(intensities)
    if 'min' in self.selection_type:
      self.buffer[index]['data'] = self.buffer[index]['data'][sorted_intensities]
      self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_intensities]
    elif 'max' in self.selection_type:
      self.buffer[index]['data'] = self.buffer[index]['data'][sorted_intensities[::-1]]
      self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_intensities[::-1]]

  def deviation_selection(self, index):
    assert self.partition_type == 'classes'

    differences = np.subtract(self.buffer[index]['data'], self.class_means[index])
    if 'abs' in self.selection_type:
      deviations = np.sum(np.abs(differences), axis=tuple(range(1, self.buffer[index]['data'].ndim)))
      sorted_deviations = np.argsort(deviations)
      if 'min' in self.selection_type:
        self.buffer[index]['data'] = self.buffer[index]['data'][sorted_deviations]
        self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_deviations]
      elif 'max' in self.selection_type:
        self.buffer[index]['data'] = self.buffer[index]['data'][sorted_deviations[::-1]]
        self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_deviations[::-1]]
    elif 'sign':
      deviations = np.sum(differences, axis=tuple(range(1, self.buffer[index]['data'].ndim)))
      sorted_deviations = np.argsort(deviations)
      half_n = int(self.buffer_size // 2) # INFO: use the upper and the lower half instead the absolute deviation
      if 'min' in self.selection_type: # FIXME: need to be indexed as cocktail shaker
        self.buffer[index]['data'] = self.buffer[index]['data'][sorted_deviations] # TODO: from inner to outer
        self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_deviations]
      elif 'max' in self.selection_type:
        self.buffer[index]['data'] = self.buffer[index]['data'][sorted_deviations] # TODO: from outer to inner
        self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_deviations]

  def logit_selection(self, index):
    stored_outputs = self.classifier(self.buffer[index]['data']).numpy() # maybe not possible if buffer is huge
    if 'class' in self.selection_type:
      buffer_measures = stored_outputs[np.arange(len(self.buffer[index]['labels'])), np.argmax(self.buffer[index]['labels'], axis=1)]
    elif 'pred' in self.selection_type:
      buffer_measures = np.ptp(stored_outputs, axis=1)

    sorted_indices = np.argsort(buffer_measures)
    if 'min' in self.selection_type:
      self.buffer[index]['data'] = self.buffer[index]['data'][sorted_indices]
      self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_indices]
    elif 'max' in self.selection_type:
      self.buffer[index]['data'] = self.buffer[index]['data'][sorted_indices[::-1]]
      self.buffer[index]['labels'] = self.buffer[index]['labels'][sorted_indices[::-1]]

  def do_selection(self, index):
    if 'last' in self.selection_type:
      pass # to get the last samples nothing have to be done
    elif 'intensity' in self.selection_type:
      self.intensity_selection(index) # need label info and compare always classwise
    elif 'deviation' in self.selection_type:
      self.deviation_selection(index) # need label info and compare always classwise
    elif 'logit' in self.selection_type:
      self.logit_selection(index) # don't care about labels -> maybe unbalanced

  def handle_mean(self, position, data):
    self.mean_counter += 1

    if len(self.class_means) == position:
      self.class_means.append(np.average(data, axis=0))
    else:
      self.class_means[position] = ((self.mean_counter - 1) * self.class_means[position] + np.average(data, axis=0)) / self.mean_counter

  def add_sort_trim(self, position, data, labels):
    if 'deviation' in self.selection_type: self.handle_mean(data)

    if len(self.buffer) == position:
      # add a new partition with data
      self.buffer.append({'data': data, 'labels': labels})

      if len(self.buffer) > 1:
        # trim old partitions beforehand (and only filled afterwards)
        number_samples = self.buffer_size // len(self.buffer)
        for index in range(len(self.buffer) - 1):
          self.buffer[index]['data'] = self.buffer[index]['data'][-number_samples:]
          self.buffer[index]['labels'] = self.buffer[index]['labels'][-number_samples:]
    else:
      # add new data to the existing partition
      self.buffer[position]['data'] = np.concatenate([self.buffer[position]['data'], data])
      self.buffer[position]['labels'] = np.concatenate([self.buffer[position]['labels'], labels])

    # sort the buffer
    self.do_selection(position)

    # trim the buffer
    number_samples = self.buffer_size // len(self.buffer)
    if len(self.buffer[position]['labels']) > number_samples:
      self.buffer[position]['data'] = self.buffer[position]['data'][-number_samples:]
      self.buffer[position]['labels'] = self.buffer[position]['labels'][-number_samples:]

  def handle_buffer(self, data, labels):
    if self.selection_type == 'reservoir':
      self.reservoir_selection(data, labels)
    else:
      if self.partition_type == 'none':
        self.add_sort_trim(0, data, labels)
      elif self.partition_type == 'tasks':
        self.add_sort_trim(self.current_task, data, labels)
      elif self.partition_type == 'classes':
        lower_class_index = self.current_task * len(self.classes[self.current_task])
        upper_class_index = lower_class_index + len(self.classes[self.current_task])

        for index, current_class in enumerate(range(lower_class_index, upper_class_index)):
          class_indices = np.where(labels[:, self.classes[self.current_task][index]] == 1)
          self.add_sort_trim(current_class, data[class_indices], labels[class_indices])

  def train_step(self, xs, ys, **kwargs):
    xs = xs.numpy()
    ys = ys.numpy()

    # handle_buffer called only with the original data and without duplicates
    # prevent side effects, like adding samples twice or to a wrong partition
    self.handle_buffer(xs, ys)

    for _ in range(self.iterations):
      # merge_batch called after handle_buffer -> not possible if in feeddict
      # prevent inconsistency for tasks regarding the number of partitions/ the number of samples per partition
      if self.current_task > 0:
        xs, ys = self.merge_batch(xs, ys)

      with tf.GradientTape() as tape:
        logits = self.classifier(xs)
        loss = self.loss_classifier(ys, logits)

      gradients = tape.gradient(loss, self.classifier.trainable_variables)
      self.optimizer_classifier.apply_gradients(zip(gradients, self.classifier.trainable_variables))

  def return_random_samples(self, number):
    # maybe crash if the buffer/partition is smaller as the number
    if self.selection_type == 'reservoir':
      indices = np.arange(self.buffer_size)
      np.random.shuffle(indices)

      data = np.array(self.buffer['data'])
      labels = np.array(self.buffer['labels'])

      return (data[indices[:number]], labels[indices[:number]])
    else:
        data = []
        labels = []

        if False: # first version: believe in random uniform
          for partition in self.buffer:
            data.append(partition['data'])
            labels.append(partition['labels'])

          data = np.concatenate(data)
          labels = np.concatenate(labels)

          indices = np.arange(labels.shape[0])
          np.random.shuffle(indices)

          return (data[indices[:number]], labels[indices[:number]])
        else: # second version: not believe in random uniform
          if self.partition_type == 'none':
            partitions = len(self.buffer)
          elif self.partition_type == 'tasks':
            partitions = len(self.buffer) - 1
          elif self.partition_type == 'classes':
            partitions = len(self.buffer) - len(self.classes[self.current_task])

          number_per_partition = number // partitions
          samples_per_partition = self.buffer_size // len(self.buffer)

          for i in range(partitions):
            indices = np.arange(samples_per_partition)
            np.random.shuffle(indices)

            # INFO: old code logic...
            # because the partitions are shrinked after the first call start counting from the end
            data.extend(self.buffer[i]['data'][indices[-number_per_partition:]])
            labels.extend(self.buffer[i]['labels'][indices[-number_per_partition:]])

          return (np.array(data), np.array(labels))

  def merge_batch(self, xs, ys):
    number_samples = int(np.round(self.batch_size * self.replay_ratio))
    buffer_data, buffer_labels = self.return_random_samples(number_samples)

    mixed_data = np.concatenate([buffer_data, xs], axis=0)
    mixed_labels = np.concatenate([buffer_labels, ys], axis=0)

    indices = np.arange(mixed_labels.shape[0])
    np.random.shuffle(indices)

    return mixed_data[indices], mixed_labels[indices]

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

    return {'NSR': metric_values}

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
