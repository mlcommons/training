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

#include <stdio.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/model/batching_model.h"
#include "cc/model/loader.h"
#include "cc/model/model.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"

// Game options flags.
DECLARE_bool(resign_enabled);
DECLARE_double(resign_threshold);
DECLARE_uint64(seed);

// Tree search flags.
DECLARE_int32(virtual_losses);
DECLARE_double(value_init_penalty);

// Inference flags.
DECLARE_string(eval_model);
DECLARE_string(eval_device);
DECLARE_int32(num_eval_readouts);

DECLARE_string(target_model);
DECLARE_string(target_device);
DECLARE_int32(num_target_readouts);

DECLARE_int32(parallel_games);

// Output flags.
DECLARE_string(output_bigtable);
DECLARE_string(sgf_dir);
DECLARE_string(bigtable_tag);
DECLARE_bool(verbose);

namespace minigo {

class Evaluator {
  class EvaluatedModel {
   public:
    EvaluatedModel(BatchingModelFactory* batcher, const std::string& path,
                   const MctsPlayer::Options& player_options)
        : batcher_(batcher), path_(path), player_options_(player_options) {}

    std::string name() {
      absl::MutexLock lock(&mutex_);
      if (name_.empty()) {
        // The model's name is lazily initialized the first time we create a
        // instance. Make sure it's valid.
        NewModelImpl();
      }
      return name_;
    }

    WinStats GetWinStats() const {
      absl::MutexLock lock(&mutex_);
      return win_stats_;
    }

    void UpdateWinStats(const Game& game) {
      absl::MutexLock lock(&mutex_);
      win_stats_.Update(game);
    }

    std::unique_ptr<Model> NewModel() {
      absl::MutexLock lock(&mutex_);
      return NewModelImpl();
    }

    const MctsPlayer::Options& player_options() const {
      return player_options_;
    }

   private:
    std::unique_ptr<Model> NewModelImpl() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
      auto model = batcher_->NewModel(path_);
      if (name_.empty()) {
        name_ = model->name();
      }
      return model;
    }

    mutable absl::Mutex mutex_;
    BatchingModelFactory* batcher_ GUARDED_BY(&mutex_);
    const std::string path_;
    std::string name_ GUARDED_BY(&mutex_);
    WinStats win_stats_ GUARDED_BY(&mutex_);
    MctsPlayer::Options player_options_;
  };

 public:
  Evaluator();

  void Reset();

  std::vector<std::pair<std::string, WinStats>> Run();

 private:
  void ThreadRun(int thread_id, EvaluatedModel* black_model,
                 EvaluatedModel* white_model);

  Game::Options game_options_;
  std::vector<std::thread> threads_;
  std::atomic<size_t> game_id_{0};
  std::vector<std::unique_ptr<BatchingModelFactory>> batchers_;
};

}  // namespace minigo

