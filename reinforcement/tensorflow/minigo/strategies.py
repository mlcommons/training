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

import copy
import math
import os
import random
import sys
import time
import sgf_wrapper

import coords
import gtp
import numpy as np
from mcts import MCTSNode, MAX_DEPTH

import go

# When to do deterministic move selection.  ~30 moves on a 19x19, ~8 on 9x9
TEMPERATURE_CUTOFF = int((go.N * go.N) / 12)


def time_recommendation(move_num, seconds_per_move=5, time_limit=15*60,
                        decay_factor=0.98):
    '''Given the current move number and the 'desired' seconds per move, return
    how much time should actually be used. This is intended specifically for
    CGOS time controls, which has an absolute 15-minute time limit.

    The strategy is to spend the maximum possible moves using seconds_per_move,
    and then switch to an exponentially decaying time usage, calibrated so that
    we have enough time for an infinite number of moves.'''

    # Divide by two since you only play half the moves in a game.
    player_move_num = move_num / 2

    # Sum of geometric series maxes out at endgame_time seconds.
    endgame_time = seconds_per_move / (1 - decay_factor)

    if endgame_time > time_limit:
        # There is so little main time that we're already in 'endgame' mode.
        base_time = time_limit * (1 - decay_factor)
        core_moves = 0
    else:
        # Leave over endgame_time seconds for the end, and play at
        # seconds_per_move for as long as possible.
        base_time = seconds_per_move
        core_moves = (time_limit - endgame_time) / seconds_per_move

    return base_time * decay_factor ** max(player_move_num - core_moves, 0)


class MCTSPlayerMixin:
    # If `simulations_per_move` is nonzero, it will perform that many reads
    # before playing. Otherwise, it uses `seconds_per_move` of wall time.
    def __init__(self, network, seconds_per_move=5, simulations_per_move=0,
                 resign_threshold=-0.90, verbosity=0, two_player_mode=False,
                 num_parallel=8):
        self.network = network
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.verbosity = verbosity
        self.two_player_mode = two_player_mode
        if two_player_mode:
            self.temp_threshold = -1
        else:
            self.temp_threshold = TEMPERATURE_CUTOFF
        self.num_parallel = num_parallel
        self.qs = []
        self.comments = []
        self.searches_pi = []
        self.root = None
        self.result = 0
        self.result_string = None
        self.resign_threshold = -abs(resign_threshold)
        super().__init__()

    def initialize_game(self, position=None):
        if position is None:
            position = go.Position()
        self.root = MCTSNode(position)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.qs = []

    def suggest_move(self, position):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()

        if self.simulations_per_move == 0:
            while time.time() - start < self.seconds_per_move:
                self.tree_search()
        else:
            current_readouts = self.root.N
            while self.root.N < current_readouts + self.simulations_per_move:
                self.tree_search()
            if self.verbosity > 0:
                print("%d: Searched %d times in %s seconds\n\n" % (
                    position.n, self.simulations_per_move, time.time() - start), file=sys.stderr)

        # print some stats on anything with probability > 1%
        if self.verbosity > 2:
            print(self.root.describe(), file=sys.stderr)
            print('\n\n', file=sys.stderr)
        if self.verbosity > 3:
            print(self.root.position, file=sys.stderr)

        return self.pick_move()

    def play_move(self, c):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''
        if not self.two_player_mode:
            self.searches_pi.append(
                self.root.children_as_pi(self.root.position.n < self.temp_threshold))
        self.qs.append(self.root.Q)  # Save our resulting Q.
        self.comments.append(self.root.describe())
        try:
            self.root = self.root.maybe_add_child(coords.to_flat(c))
        except go.IllegalMove:
            print("Illegal move")
            if not self.two_player_mode:
                self.searches_pi.pop()
            self.qs.pop()
            self.comments.pop()
            return False
        self.position = self.root.position  # for showboard
        del self.root.parent.children
        return True  # GTP requires positive result.

    def pick_move(self):
        '''Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.position.n > self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return coords.from_flat(fcoord)

    def tree_search(self, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            leaf = self.root.select_leaf()
            if self.verbosity >= 4:
                print(self.show_path_to_root(leaf))
            # if game is over, override the value estimate with the true score
            if leaf.is_done():
                value = 1 if leaf.position.score() > 0 else -1
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.network.run_many(
                [leaf.position for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)

    def show_path_to_root(self, node):
        pos = node.position
        diff = node.position.n - self.root.position.n
        if len(pos.recent) == 0:
            return

        def fmt(move): return "{}-{}".format('b' if move.color == 1 else 'w',
                                             coords.to_kgs(move.move))
        path = " ".join(fmt(move) for move in pos.recent[-diff:])
        if node.position.n >= MAX_DEPTH:
            path += " (depth cutoff reached) %0.1f" % node.position.score()
        elif node.position.is_game_over():
            path += " (game over) %0.1f" % node.position.score()
        return path

    def should_resign(self):
        '''Returns true if the player resigned.  No further moves should be played'''
        return self.root.Q_perspective < self.resign_threshold

    def set_result(self, winner, was_resign):
        self.result = winner
        if was_resign:
            string = "B+R" if winner == go.BLACK else "W+R"
        else:
            string = self.root.position.result_string()
        self.result_string = string

    def to_sgf(self, use_comments=True):
        assert self.result_string is not None
        pos = self.root.position
        if use_comments:
            comments = self.comments or ['No comments.']
            comments[0] = ("Resign Threshold: %0.3f\n" %
                           self.resign_threshold) + comments[0]
        else:
            comments = []
        return sgf_wrapper.make_sgf(pos.recent, self.result_string,
                                    white_name=os.path.basename(
                                        self.network.save_file) or "Unknown",
                                    black_name=os.path.basename(
                                        self.network.save_file) or "Unknown",
                                    comments=comments)

    def is_done(self):
        return self.result != 0 or self.root.is_done()

    def extract_data(self):
        assert len(self.searches_pi) == self.root.position.n
        assert self.result != 0
        for pwc, pi in zip(go.replay_position(self.root.position, self.result),
                           self.searches_pi):
            yield pwc.position, pi, pwc.result

    def chat(self, msg_type, sender, text):
        default_response = "Supported commands are 'winrate', 'nextplay', 'fortune', and 'help'."
        if self.root is None or self.root.position.n == 0:
            return "I'm not playing right now.  " + default_response

        if 'winrate' in text.lower():
            wr = (abs(self.root.Q) + 1.0) / 2.0
            color = "Black" if self.root.Q > 0 else "White"
            return "{:s} {:.2f}%".format(color, wr * 100.0)
        elif 'nextplay' in text.lower():
            return "I'm thinking... " + self.root.most_visited_path()
        elif 'fortune' in text.lower():
            return "You're feeling lucky!"
        elif 'help' in text.lower():
            return "I can't help much with go -- try ladders!  Otherwise: " + default_response
        else:
            return default_response


class CGOSPlayerMixin(MCTSPlayerMixin):
    def suggest_move(self, position):
        self.seconds_per_move = time_recommendation(position.n)
        return super().suggest_move(position)
