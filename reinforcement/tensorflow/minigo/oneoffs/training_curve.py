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

"""
Used to plot the accuracy of the policy and value networks in
predicting professional game moves and results over the course
of training. Check FLAGS for default values for what models to
load and what sgf files to parse.

Usage:
python training_curve.py

Sample 3 positions from each game
python training_curve.py --num_positions=3

Only grab games after 2005 (default is 2000)
python training_curve.py --min_year=2005
"""
import sys
sys.path.insert(0, '.')

import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags
from tqdm import tqdm

import coords
from rl_loop import fsdb
import oneoff_utils

flags.DEFINE_string("sgf_dir", None, "sgf database")

flags.DEFINE_string("plot_dir", "data", "Where to save the plots.")
flags.DEFINE_integer("min_year", "2000",
                     "Only take sgf games with date >= min_year")
flags.DEFINE_string("komi", "7.5", "Only take sgf games with given komi")
flags.DEFINE_integer("idx_start", 150, "Only take models after given idx")
flags.DEFINE_integer("num_positions", 1,
                     "How many positions from each game to sample from.")
flags.DEFINE_integer("eval_every", 5,
                     "Eval every k models to generate the curve")

flags.mark_flag_as_required('sgf_dir')

FLAGS = flags.FLAGS


def batch_run_many(player, positions, batch_size=100):
    """Used to avoid a memory oveflow issue when running the network
    on too many positions. TODO: This should be a member function of
    player.network?"""
    prob_list = []
    value_list = []
    for idx in range(0, len(positions), batch_size):
        probs, values = player.network.run_many(positions[idx:idx + batch_size])
        prob_list.append(probs)
        value_list.append(values)
    return np.concatenate(prob_list, axis=0), np.concatenate(value_list, axis=0)


def eval_player(player, positions, moves, results):
    probs, values = batch_run_many(player, positions)
    policy_moves = [coords.from_flat(c) for c in np.argmax(probs, axis=1)]
    top_move_agree = [moves[idx] == policy_moves[idx]
                      for idx in range(len(moves))]
    square_err = (values - results) ** 2 / 4
    return top_move_agree, square_err


def sample_positions_from_games(sgf_files, num_positions=1):
    pos_data = []
    move_data = []
    result_data = []
    move_idxs = []

    fail_count = 0
    for path in tqdm(sgf_files, desc="loading sgfs", unit="games"):
        try:
            positions, moves, results = oneoff_utils.parse_sgf_to_examples(path)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("Parse exception:", e)
            fail_count += 1
            continue

        # add entire game
        if num_positions == -1:
            pos_data.extend(positions)
            move_data.extend(moves)
            move_idxs.extend(range(len(positions)))
            result_data.extend(results)
        else:
            for idx in np.random.choice(len(positions), num_positions):
                pos_data.append(positions[idx])
                move_data.append(moves[idx])
                result_data.append(results[idx])
                move_idxs.append(idx)
    print("Sampled {} positions, failed to parse {} files".format(
        len(pos_data), fail_count))
    return pos_data, move_data, result_data, move_idxs


def get_training_curve_data(
        model_dir, pos_data, move_data, result_data, idx_start, eval_every):
    model_paths = oneoff_utils.get_model_paths(model_dir)
    df = pd.DataFrame()
    player = None

    print("Evaluating models {}-{}, eval_every={}".format(
        idx_start, len(model_paths), eval_every))
    for idx in tqdm(range(idx_start, len(model_paths), eval_every)):
        if player:
            oneoff_utils.restore_params(model_paths[idx], player)
        else:
            player = oneoff_utils.load_player(model_paths[idx])

        correct, squared_errors = eval_player(
            player=player, positions=pos_data,
            moves=move_data, results=result_data)

        avg_acc = np.mean(correct)
        avg_mse = np.mean(squared_errors)
        print("Model: {}, acc: {:.4f}, mse: {:.4f}".format(
            model_paths[idx], avg_acc, avg_mse))
        df = df.append({"num": idx, "acc": avg_acc,
                        "mse": avg_mse}, ignore_index=True)
    return df


def save_plots(data_dir, df):
    plt.plot(df["num"], df["acc"])
    plt.xlabel("Model idx")
    plt.ylabel("Accuracy")
    plt.title("Accuracy in Predicting Professional Moves")
    plot_path = os.path.join(data_dir, "move_acc.pdf")
    plt.savefig(plot_path)

    plt.figure()

    plt.plot(df["num"], df["mse"])
    plt.xlabel("Model idx")
    plt.ylabel("MSE/4")
    plt.title("MSE in predicting outcome")
    plot_path = os.path.join(data_dir, "value_mse.pdf")
    plt.savefig(plot_path)


def main(unusedargv):
    sgf_files = oneoff_utils.find_and_filter_sgf_files(
        FLAGS.sgf_dir, FLAGS.min_year, FLAGS.komi)
    pos_data, move_data, result_data, move_idxs = sample_positions_from_games(
        sgf_files=sgf_files, num_positions=FLAGS.num_positions)
    df = get_training_curve_data(fsdb.models_dir(), pos_data, move_data,
                                 result_data, FLAGS.idx_start, FLAGS.eval_every)
    save_plots(FLAGS.plot_dir, df)


if __name__ == "__main__":
    app.run(main)
