# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import yaml

from common.data.dali.pipeline import PipelineParams, SpeedPerturbationParams
from common.data.text import Tokenizer
from common.data import features
from common.helpers import print_once
from .model import RNNT


def default_args(klass):
    sig = inspect.signature(klass.__init__)
    return {k: v.default for k,v in sig.parameters.items() if k != 'self'}


def load(fpath, max_duration=None):

    if fpath.endswith('.toml'):
        raise ValueError('.toml config format has been changed to .yaml')

    cfg = yaml.safe_load(open(fpath, 'r'))

    # Reload to deep copy shallow copies, which were made with yaml anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    cfg = yaml.safe_load(yaml.dump(cfg))

    # Modify the config with supported cmdline flags
    if max_duration is not None:
        cfg['input_train']['audio_dataset']['max_duration'] = max_duration
        cfg['input_train']['filterbank_features']['max_duration'] = max_duration

    return cfg


def validate_and_fill(klass, user_conf, ignore=[], optional=[]):
    conf = default_args(klass)

    for k,v in user_conf.items():
        assert k in conf or k in ignore, f'Unknown parameter {k} for {klass}'
        conf[k] = v

    # Keep only mandatory or optional-nonempty
    conf = {k:v for k,v in conf.items()
            if k not in optional or v is not inspect.Parameter.empty}

    # Validate
    for k,v in conf.items():
        assert v is not inspect.Parameter.empty, \
            f'Value for {k} not specified for {klass}'
    return conf


def input(conf_yaml, split='train'):

    conf = copy.deepcopy(conf_yaml[f'input_{split}'])
    conf_dataset = conf.pop('audio_dataset')
    conf_features = conf.pop('filterbank_features')
    conf_splicing = conf.pop('frame_splicing', {})
    conf_specaugm = conf.pop('spec_augment', None)
    conf_cutoutau = conf.pop('cutout_augment', None)

    # Validate known inner classes
    inner_classes = [
        (conf_dataset, 'speed_perturbation', SpeedPerturbationParams),
    ]
    amp=['optim_level']
    for conf_tgt, key, klass in inner_classes:
        if key in conf_tgt:
            conf_tgt[key] = validate_and_fill(klass, conf_tgt[key], optional=amp)

    for k in conf:
        raise ValueError(f'Unknown key {k}')

    # Validate outer classes
    conf_dataset = validate_and_fill(PipelineParams, conf_dataset)

    conf_features = validate_and_fill(features.FilterbankFeatures, conf_features, optional=amp)
    conf_splicing = validate_and_fill(features.FrameSplicing, conf_splicing, optional=amp)
    conf_specaugm = conf_specaugm and validate_and_fill(features.SpecAugment, conf_specaugm, optional=amp)

    # Check params shared between classes
    for shared in ['sample_rate']:
        assert conf_dataset[shared] == conf_features[shared], (
            f'{shared} should match in Dataset and FeatureProcessor: '
            f'{conf_dataset[shared]}, {conf_features[shared]}')

    return conf_dataset, conf_features, conf_splicing, conf_specaugm


def rnnt(conf):
    return validate_and_fill(RNNT, conf['rnnt'], optional=['n_classes'])


def tokenizer(conf):
    return validate_and_fill(Tokenizer, conf['tokenizer'], optional=['sentpiece_model'])


def apply_duration_flags(cfg, max_duration):
    if max_duration is not None:
        cfg['input_train']['audio_dataset']['max_duration'] = max_duration
        cfg['input_train']['filterbank_features']['max_duration'] = max_duration

