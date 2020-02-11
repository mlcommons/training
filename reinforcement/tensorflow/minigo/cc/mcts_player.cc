// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/mcts_player.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "cc/logging.h"
#include "cc/random.h"

namespace minigo {

std::ostream& operator<<(std::ostream& os, const MctsPlayer::Options& options) {
  os << options.tree << " inject_noise:" << options.inject_noise
     << " virtual_losses:" << options.virtual_losses
     << " num_readouts:" << options.num_readouts
     << " seconds_per_move:" << options.seconds_per_move
     << " time_limit:" << options.time_limit
     << " decay_factor:" << options.decay_factor
     << " fastplay_frequency:" << options.fastplay_frequency
     << " fastplay_readouts:" << options.fastplay_readouts
     << " target_pruning:" << options.target_pruning
     << " random_seed:" << options.random_seed << std::flush;
  return os;
}

float TimeRecommendation(int move_num, float seconds_per_move, float time_limit,
                         float decay_factor) {
  // Divide by two since you only play half the moves in a game.
  int player_move_num = move_num / 2;

  // Sum of geometric series maxes out at endgame_time seconds.
  float endgame_time = seconds_per_move / (1.0f - decay_factor);

  float base_time;
  int core_moves;
  if (endgame_time > time_limit) {
    // There is so little main time that we're already in 'endgame' mode.
    base_time = time_limit * (1.0f - decay_factor);
    core_moves = 0;
  } else {
    // Leave over endgame_time seconds for the end, and play at
    // seconds_per_move for as long as possible.
    base_time = seconds_per_move;
    core_moves = (time_limit - endgame_time) / seconds_per_move;
  }

  return base_time *
         std::pow(decay_factor, std::max(player_move_num - core_moves, 0));
}

MctsPlayer::MctsPlayer(std::unique_ptr<Model> model,
                       std::shared_ptr<InferenceCache> inference_cache,
                       Game* game, const Options& options)
    : model_(std::move(model)),
      game_(game),
      rnd_(options.random_seed, Random::kUniqueStream),
      options_(options),
      inference_cache_(std::move(inference_cache)),
      inference_mix_(rnd_.UniformUint64()) {
  NewGame();
}

MctsPlayer::~MctsPlayer() = default;

void MctsPlayer::InitializeGame(const Position& position) {
  tree_ = absl::make_unique<MctsTree>(position, options_.tree);
  game_->NewGame();
}

void MctsPlayer::NewGame() { InitializeGame(Position(Color::kBlack)); }

bool MctsPlayer::UndoMove() {
  if (!tree_->UndoMove()) {
    return false;
  }
  game_->UndoMove();
  return true;
}

Coord MctsPlayer::SuggestMove(int new_readouts, bool inject_noise) {
  auto start = absl::Now();

  if (inject_noise) {
    InjectNoise(kDirichletAlpha);
  }

  const auto* root = tree_->root();
  int target_readouts = root->N() + new_readouts;

  if (options_.seconds_per_move > 0) {
    // Use time to limit the number of reads.
    float seconds_per_move = options_.seconds_per_move;
    if (options_.time_limit > 0) {
      seconds_per_move =
          TimeRecommendation(root->position.n(), seconds_per_move,
                             options_.time_limit, options_.decay_factor);
    }
    while (absl::Now() - start < absl::Seconds(seconds_per_move)) {
      TreeSearch(options_.virtual_losses, target_readouts);
    }
  } else {
    // Use a fixed number of reads.
    while (root->N() < target_readouts) {
      TreeSearch(options_.virtual_losses, target_readouts);
    }
  }
  if (ShouldResign()) {
    return Coord::kResign;
  }

  return tree_->PickMove(&rnd_, true);
}

void MctsPlayer::TreeSearch(int num_leaves, int max_num_reads) {
  MaybeExpandRoot();
  SelectLeaves(num_leaves, max_num_reads);
  ProcessLeaves();
}

void MctsPlayer::InjectNoise(float dirichlet_alpha) {
  MaybeExpandRoot();
  tree_->InjectNoise(rnd_.Dirichlet<kNumMoves>(kDirichletAlpha),
                     options_.noise_mix);
}

void MctsPlayer::MaybeExpandRoot() {
  if (!tree_->root()->is_expanded) {
    SelectLeaves(1, tree_->root()->N() + 1);
    ProcessLeaves();
  }
}

void MctsPlayer::SelectLeaves(int num_leaves, int max_num_reads) {
  tree_search_inferences_.clear();
  ModelOutput cached_output;

  int max_cache_misses = num_leaves * 2;
  int num_selected = 0;
  int num_cache_misses = 0;
  while (num_cache_misses < max_cache_misses &&
         tree_->root()->N() < max_num_reads) {
    auto* leaf = tree_->SelectLeaf(true);

    if (leaf->game_over()) {
      float value =
          leaf->position.CalculateScore(game_->options().komi) > 0 ? 1 : -1;
      tree_->IncorporateEndGameResult(leaf, value);
      ++num_cache_misses;
      continue;
    }

    // Calculate the symmetry we want to use for inference.
    auto inference_sym = GetInferenceSymmetry(leaf);
    auto canonical_sym = GetCanonicalSymmetry(leaf);

    InferenceCache::Key cache_key;
    if (inference_cache_ != nullptr) {
      cache_key =
          InferenceCache::Key(leaf->move, canonical_sym, leaf->position);

      if (inference_cache_->TryGet(cache_key, canonical_sym, inference_sym,
                                   &cached_output)) {
        tree_->IncorporateResults(leaf, cached_output.policy,
                                  cached_output.value);
        continue;
      }
    }

    ++num_cache_misses;

    tree_search_inferences_.emplace_back(cache_key, canonical_sym,
                                         inference_sym, leaf);

    auto& input = tree_search_inferences_.back().input;
    input.sym = inference_sym;
    // TODO(tommadams): add a method to Model that returns the required position
    // history size.
    auto* node = leaf;
    for (int i = 0; i < input.position_history.capacity(); ++i) {
      input.position_history.push_back(&node->position);
      node = node->parent;
      if (node == nullptr) {
        break;
      }
    }

    tree_->AddVirtualLoss(leaf);
    if (++num_selected == num_leaves) {
      // We found enough leaves.
      break;
    }
    if (leaf == tree_->root()) {
      // If the root is a leaf, we can't possibly find any other leaves.
      break;
    }
  }
}

bool MctsPlayer::ShouldResign() const {
  return game_->options().resign_enabled &&
         tree_->root()->Q_perspective() < game_->options().resign_threshold;
}

void MctsPlayer::SetTreeSearchCallback(TreeSearchCallback cb) {
  tree_search_cb_ = std::move(cb);
}

std::string MctsPlayer::GetModelsUsedForInference() const {
  std::vector<std::string> parts;
  parts.reserve(inferences_.size());
  for (const auto& info : inferences_) {
    parts.push_back(absl::StrCat(info.model, "(", info.first_move, ",",
                                 info.last_move, ")"));
  }
  return absl::StrJoin(parts, ", ");
}

bool MctsPlayer::PlayMove(Coord c, bool is_trainable) {
  if (tree_->is_game_over()) {
    MG_LOG(ERROR) << "Can't play move " << c << ", game is over";
    return false;
  }

  // Handle resignations.
  if (c == Coord::kResign) {
    game_->SetGameOverBecauseOfResign(OtherColor(tree_->to_play()));
    return true;
  }

  if (!tree_->is_legal_move(c)) {
    MG_LOG(ERROR) << "Move " << c << " is illegal";
    // We're probably about to crash. Dump the player's options and the moves
    // that got us to this point.
    MG_LOG(ERROR) << "MctsPlayer options: " << options_;
    MG_LOG(ERROR) << "Game options: " << game_->options();
    for (int i = 0; i < game_->num_moves(); ++i) {
      const auto* move = game_->GetMove(i);
      MG_LOG(ERROR) << move->color << "  " << move->c;
    }
    return false;
  }

  // Adjust the visits before adding the move's search_pi to the Game.
  if (is_trainable && options_.target_pruning) {
    tree_->ReshapeFinalVisits(true);
  }

  UpdateGame(c, is_trainable);

  tree_->PlayMove(c);

  // Handle consecutive passing or termination by move limit.
  if (tree_->is_game_over()) {
    game_->SetGameOverBecauseOfPasses(
        tree_->CalculateScore(game_->options().komi));
  }

  return true;
}

void MctsPlayer::PlayOpponentsMove(Coord c) {
  if (!tree_->is_game_over()) {
    MG_CHECK(c != Coord::kResign);
    MG_CHECK(tree_->is_legal_move(c));
    tree_->PlayMove(c);
  }
}

void MctsPlayer::UpdateGame(Coord c, bool is_trainable) {
  const auto* root = tree_->root();

  // Record which model(s) were used when running tree search for this move.
  std::vector<std::string> models;
  if (!inferences_.empty()) {
    for (auto it = inferences_.rbegin(); it != inferences_.rend(); ++it) {
      if (it->last_move < root->position.n()) {
        break;
      }
      models.push_back(it->model);
    }
    std::reverse(models.begin(), models.end());
  }

  // Build a comment for the move.
  auto comment = tree_->Describe();
  if (!models.empty()) {
    comment =
        absl::StrCat("models:", absl::StrJoin(models, ","), "\n", comment);
  }

  // Update the game history.
  if (is_trainable && c != Coord::kResign) {
    auto search_pi = tree_->CalculateSearchPi();
    game_->AddTrainableMove(tree_->to_play(), c, root->position,
                            std::move(comment), root->Q(), root->N(), search_pi);
  } else {
    game_->AddNonTrainableMove(tree_->to_play(), c, root->position,
                               std::move(comment), root->Q(), root->N());
  }
}

// TODO(tommadams): move this up to below SelectLeaves.
void MctsPlayer::ProcessLeaves() {
  if (tree_search_inferences_.empty()) {
    return;
  }

  input_ptrs_.clear();
  output_ptrs_.clear();
  for (auto& x : tree_search_inferences_) {
    input_ptrs_.push_back(&x.input);
    output_ptrs_.push_back(&x.output);
  }

  // Run inference.
  model_->RunMany(input_ptrs_, &output_ptrs_, &inference_model_);

  // Record some information about the inference.
  if (!inference_model_.empty()) {
    if (inferences_.empty() || inference_model_ != inferences_.back().model) {
      inferences_.emplace_back(inference_model_, tree_->root()->position.n());
    }
    inferences_.back().last_move = tree_->root()->position.n();
    inferences_.back().total_count += tree_search_inferences_.size();
  }

  // Incorporate the inference outputs back into tree search.
  for (auto& inference : tree_search_inferences_) {
    auto& output = inference.output;

    // Merge the inference output with those in the inference cache, possibly
    // updating the values in `output`.
    if (inference_cache_ != nullptr) {
      inference_cache_->Merge(inference.cache_key, inference.canonical_sym,
                              inference.inference_sym, &output);
    }

    // Propagate the results back up the tree to the root.
    tree_->IncorporateResults(inference.leaf, output.policy, output.value);
    tree_->RevertVirtualLoss(inference.leaf);
  }

  if (tree_search_cb_ != nullptr) {
    std::vector<const MctsNode*> leaves;
    leaves.reserve(tree_search_inferences_.size());
    for (auto& inference : tree_search_inferences_) {
      leaves.push_back(inference.leaf);
    }
    tree_search_cb_(leaves);
  }
}

}  // namespace minigo
