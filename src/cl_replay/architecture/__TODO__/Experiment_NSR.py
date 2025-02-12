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
import time
import numpy           as np
import tensorflow      as tf

from cl_replay.utils             import log
from cl_replay.experiments        import Experiment

class Experiment_NSR(Experiment):

    def _init_parser(self):

        Experiment._init_parser(self)
        self.buffer_size    = self.parser.add_argument('--buffer_size'   , type=int  , default=1000                                        , help='the size of the replay buffer')
        self.replay_ratio   = self.parser.add_argument('--replay_ratio'  , type=float, default=1.0                                         , help='the ratio between old and new samples')
        self.balance_type   = self.parser.add_argument('--balance_type'  , type=str  , default='none', choices=['none', 'tasks', 'classes'], help='how should the data be balanced? [default="none", "tasks", "classes"]')
        self.partition_type = self.parser.add_argument('--partition_type', type=str  , default='none', choices=['none', 'tasks', 'classes'], help='how should the buffer be balanced? [default="none", "tasks", "classes"]')
        self.selection_type = self.parser.add_argument('--selection_type', type=str  , default='last', choices=['last', 'reservoir', 'min_intensity', 'max_intensity', 'abs_min_deviation', 'abs_max_deviation', 'sing_min_deviation', 'sing_max_deviation', 'class_min_logit', 'class_max_logit', 'pred_min_logit', 'pred_max_logit'],
                                                       help='how should samples be mixed? [default="last", "reservoir", "min_intensity", "max_intensity", "abs_min_deviation", "abs_max_deviation", "sing_min_deviation", "sing_max_deviation", "class_min_logit", "class_max_logit", "pred_min_logit", "pred_max_logit"]')
        self.drop_model     = self.parser.add_argument('--drop_model'    , type=str  , default='no'  , choices=['no', 'yes']               , help='should the model be dropped every task? [default="no", "yes"]')
        self.model_type     = self.parser.add_argument('--model_type'    , type=str  , default='NSR'                                       , help='class to load form module "model"')

    #TODO: Why does this call fit by itself?, why not set model like other experiment classes do
    def fit(self, current_task, **kwargs):
        return self.model.fit(**kwargs)


    def before_task(self, current_task, **kwargs):
        # FIXME: classes are strings, but there is no mapping to names (missing lookup)?!
        self.model.classes.append([int(class_) for class_ in self.tasks[self.task]])

        if self.drop_model == 'yes':
            self.model.reset_model(current_task)


    def after_task(self, current_task, **kwargs):
        #self.model.current_task += 1 #TODO: this should be automated in Experiment.py / Experiment_Replay.py
        pass


if __name__ == '__main__':
    Experiment_NSR().train()
