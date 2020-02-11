# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from tensorflow import gfile
from tqdm import tqdm

import dual_net
from rl_loop import shipname
from strategies import MCTSPlayer
import sgf_wrapper
from utils import logged_timer


def final_position_sgf(sgf_path):
    for pwc in sgf_wrapper.replay_sgf_file(sgf_path):
        pass

    return pwc.position.play_move(pwc.next_move)


def parse_sgf_to_examples(sgf_path):
    """Return supervised examples from positions

    NOTE: last move is not played because no p.next_move after.
    """

    return zip(*[(p.position, p.next_move, p.result)
                 for p in sgf_wrapper.replay_sgf_file(sgf_path)])


def check_year(props, year):
    if year is None:
        return True
    if props.get('DT') is None:
        return False

    try:
        # Most sgf files in this database have dates of the form
        # "2005-01-15", but there are some rare exceptions like
        # "Broadcasted on 2005-01-15.
        year_sgf = int(props.get('DT')[0][:4])
    except:
        return False
    return year_sgf >= year


def check_komi(props, komi_str):
    if komi_str is None:
        return True
    if props.get('KM') is None:
        return False
    return props.get('KM')[0] == komi_str


def filter_year_komi(min_year=None, komi=None):
    def validate(path):
        with open(path) as f:
            sgf_contents = f.read()
        props = sgf_wrapper.get_sgf_root_node(sgf_contents).properties
        return check_komi(props, komi) and check_year(props, min_year)
    return validate


def find_and_filter_sgf_files(base_dir, min_year=None, komi=None):
    """Finds all sgf files in base_dir with year >= min_year and komi"""
    sgf_files = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.sgf'):
                path = os.path.join(dirpath, filename)
                sgf_files.append(path)

    if min_year == komi == None:
        print ("Found {} sgf_files".format(len(sgf_files)))
        return sgf_files

    f = filter_year_komi(min_year, komi)
    filtered_sgf_files = [sgf for sgf in tqdm(sgf_files) if f(sgf)]

    print("{} of {} .sgf files matched (min_year >= {}, komi = {})".format(
        len(filtered_sgf_files), len(sgf_files), min_year, komi))
    return filtered_sgf_files


def get_model_paths(model_dir):
    """Returns all model paths in the model_dir."""
    all_models = gfile.Glob(os.path.join(model_dir, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = [
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames]
    model_names = sorted(model_numbers_names)
    return [os.path.join(model_dir, name[1]) for name in model_names]


def load_player(model_path):
    print("Loading weights from %s ... " % model_path)
    with logged_timer("Loading weights from %s ... " % model_path):
        network = dual_net.DualNetwork(model_path)
        network.name = os.path.basename(model_path)
    player = MCTSPlayer(network)
    return player


def restore_params(model_path, player):
    with player.network.sess.graph.as_default():
        player.network.initialize_weights(model_path)
