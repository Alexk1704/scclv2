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
import sys
import numpy as np
import tensorflow as tf

from cl_replay.utils import log
from Experiment import Experiment


class Experiment_GEM(Experiment):
    def _init_parser(self):
        Experiment._init_parser(self)
        self.buffer_size    = self.parser.add_argument('--buffer_size',     type=int, default=1000,     help='the size of the replay buffer')
        self.balance_type   = self.parser.add_argument('--balance_type',    type=str, default='none',   choices=['none', 'tasks', 'classes'], help='how should the data be balanced? [default="none", "tasks", "classes"]')
        self.selection_type = self.parser.add_argument('--selection_type',  type=int, default='last',   choices=['last', 'random'], help='how should samples be selected? [default="last", "random"]')
        self.drop_model     = self.parser.add_argument('--drop_model',      type=str, default='no',     choices=['no', 'yes'], help='should the model be dropped every task? [default="no", "yes"]')
        self.model_type     = self.parser.add_argument('--model_type',      type=str, default='GEM',    help='class to load form module "model"')

    def before_task(self, current_task, **kwargs):
        if self.drop_model == 'yes': self.model.reset_model(current_task)

    def after_task(self, current_task, **kwargs):
        #self.model.current_task += 1
        pass

if __name__ == '__main__':
    Experiment_GEM().train()
