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

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "cc/algorithm.h"
#include "cc/color.h"
#include "cc/constants.h"
#include "cc/dual_net/fake_dual_net.h"
#include "cc/position.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

namespace minigo {
namespace {

static constexpr char kAlmostDoneBoard[] = R"(
    .XO.XO.OO
    X.XXOOOO.
    XXXXXOOOO
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    .XXXXOOO.
    XXXXXOOOO
    XXXXOOOOO)";

// Tromp taylor means black can win if we hit the move limit.
static constexpr char kTtFtwBoard[] = R"(
    .XXOOOOOO
    X.XOO...O
    .XXOO...O
    X.XOO...O
    .XXOO..OO
    X.XOOOOOO
    .XXOOOOOO
    X.XXXXXXX
    XXXXXXXXX)";

static constexpr char kOneStoneBoard[] = R"(
    .........
    .........
    .........
    .........
    ....X....
    .........
    .........
    .........
    .........)";

int CountPendingVirtualLosses(const MctsNode* node) {
  int num = 0;
  std::vector<const MctsNode*> pending{node};
  while (!pending.empty()) {
    node = pending.back();
    pending.pop_back();
    MG_CHECK(node->num_virtual_losses_applied >= 0);
    num += node->num_virtual_losses_applied;
    for (const auto& p : node->children) {
      pending.push_back(p.second.get());
    }
  }
  return num;
}

class TestablePlayer : public MctsPlayer {
 public:
  explicit TestablePlayer(Game* game, const MctsPlayer::Options& player_options)
      : MctsPlayer(absl::make_unique<FakeDualNet>(), nullptr, game,
                   player_options) {}

  explicit TestablePlayer(std::unique_ptr<Model> model, Game* game,
                          const Options& options)
      : MctsPlayer(std::move(model), nullptr, game, options) {}

  TestablePlayer(absl::Span<const float> fake_priors, float fake_value,
                 Game* game, const Options& options)
      : MctsPlayer(absl::make_unique<FakeDualNet>(fake_priors, fake_value),
                   nullptr, game, options) {}

  using MctsPlayer::mutable_tree;
  using MctsPlayer::PlayMove;
  using MctsPlayer::TreeSearch;
  using MctsPlayer::UndoMove;

  ModelOutput Run(const ModelInput& input) {
    ModelOutput output;
    std::vector<const ModelInput*> inputs = {&input};
    std::vector<ModelOutput*> outputs = {&output};
    model()->RunMany(inputs, &outputs, nullptr);
    return output;
  }
};

class MctsPlayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Game::Options game_options;
    game_ = absl::make_unique<Game>("b", "w", game_options);
  }

  std::unique_ptr<TestablePlayer> CreateBasicPlayer(
      MctsPlayer::Options player_options) {
    // Always use a deterministic random seed.
    player_options.random_seed = 17;

    auto player =
        absl::make_unique<TestablePlayer>(game_.get(), player_options);
    auto* tree = player->mutable_tree();

    ModelInput input;
    input.position_history.push_back(&player->root()->position);
    auto output = player->Run(input);
    tree->IncorporateResults(tree->SelectLeaf(true), output.policy,
                             output.value);
    return player;
  }

  std::unique_ptr<TestablePlayer> CreateAlmostDonePlayer() {
    Game::Options game_options;
    game_options.komi = 2.5;
    game_ = absl::make_unique<Game>("b", "w", game_options);

    // Always use a deterministic random seed.
    MctsPlayer::Options player_options;
    player_options.random_seed = 17;

    std::array<float, kNumMoves> probs;
    for (auto& p : probs) {
      p = 0.001;
    }
    probs[Coord(0, 2)] = 0.2;
    probs[Coord(0, 3)] = 0.2;
    probs[Coord(0, 4)] = 0.2;
    probs[Coord::kPass] = 0.2;

    auto player = absl::make_unique<TestablePlayer>(probs, 0, game_.get(),
                                                    player_options);
    auto board = TestablePosition(kAlmostDoneBoard, Color::kBlack);
    player->InitializeGame(board);
    return player;
  }

  std::unique_ptr<Game> game_;
};

TEST_F(MctsPlayerTest, TimeRecommendation) {
  // Early in the game with plenty of time left, the time recommendation should
  // be the requested number of seconds per move.
  EXPECT_EQ(5, TimeRecommendation(0, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(1, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(10, 5, 1000, 0.98));
  EXPECT_EQ(5, TimeRecommendation(50, 5, 1000, 0.98));

  // With a small time limit, the time recommendation should immediately be less
  // than requested.
  EXPECT_GT(1.0f, TimeRecommendation(0, 5, 10, 0.98));

  // Time recommendations for even and odd moves should be identical.
  EXPECT_EQ(TimeRecommendation(20, 5, 10, 0.98),
            TimeRecommendation(21, 5, 10, 0.98));

  // If we're later into the game than should really be possible, time
  // recommendation should be almost zero.
  EXPECT_GT(0.0001, TimeRecommendation(1000, 5, 100, 0.98));
}

TEST_F(MctsPlayerTest, DontPassIfLosing) {
  auto player = CreateAlmostDonePlayer();
  auto* root = player->root();
  EXPECT_EQ(-0.5, root->position.CalculateScore(game_->options().komi));

  for (int i = 0; i < 20; ++i) {
    player->TreeSearch(1, std::numeric_limits<int>::max());
  }

  // Search should converge on D9 as only winning move.
  auto best_move = ArgMax(root->edges.N);
  ASSERT_EQ(Coord::FromGtp("D9"), best_move);
  // D9 should have a positive value.
  EXPECT_LT(0, root->child_Q(best_move));
  EXPECT_LE(20, root->N());
  // Passing should be ineffective.
  EXPECT_GT(0, root->child_Q(Coord::kPass));

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST_F(MctsPlayerTest, ParallelTreeSearch) {
  auto player = CreateAlmostDonePlayer();
  auto* root = player->root();

  // Initialize the tree so that the root node has populated children.
  player->TreeSearch(1, std::numeric_limits<int>::max());
  // Virtual losses should enable multiple searches to happen simultaneously
  // without throwing an error...
  for (int i = 0; i < 5; ++i) {
    player->TreeSearch(5, std::numeric_limits<int>::max());
  }

  // Search should converge on D9 as only winning move.
  auto best_move = ArgMax(root->edges.N);
  EXPECT_EQ(Coord::FromString("D9"), best_move);
  // D9 should have a positive value.
  EXPECT_LT(0, root->child_Q(best_move));
  EXPECT_LE(20, root->N());
  // Passing should be ineffective.
  EXPECT_GT(0, root->child_Q(Coord::kPass));

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST_F(MctsPlayerTest, RidiculouslyParallelTreeSearch) {
  auto player = CreateAlmostDonePlayer();
  auto* root = player->root();

  for (int i = 0; i < 10; ++i) {
    // Test that an almost complete game will tree search with
    // # parallelism > # legal moves.
    player->TreeSearch(50, std::numeric_limits<int>::max());
  }

  // No virtual losses should be pending.
  EXPECT_EQ(0, CountPendingVirtualLosses(root));
}

TEST_F(MctsPlayerTest, ColdStartParallelTreeSearch) {
  MctsPlayer::Options options;
  options.random_seed = 17;
  auto player = absl::make_unique<TestablePlayer>(absl::Span<const float>(),
                                                  0.17, game_.get(), options);
  const auto* root = player->root();

  // Test that parallel tree search doesn't trip on an empty tree.
  EXPECT_EQ(0, root->N());
  EXPECT_EQ(false, root->is_expanded);
  player->TreeSearch(4, std::numeric_limits<int>::max());
  EXPECT_EQ(0, CountPendingVirtualLosses(root));

  // The TreeSearch(4) call will have first expanded the root node so that it
  // can perform the requested search for a total of 5 visits.
  EXPECT_EQ(5, root->N());

  // 0.14167 = average(0, 0.17) / (N + 1), since 0 is the prior on the root.
  EXPECT_NEAR(0.14167, root->Q(), 0.001) << root->W() << " : " << root->N();
}

TEST_F(MctsPlayerTest, TreeSearchFailsafe) {
  // Test that the failsafe works correctly. It can trigger if the MCTS
  // repeatedly visits a finished game state.
  std::array<float, kNumMoves> probs;
  for (auto& p : probs) {
    p = 0.001;
  }
  probs[Coord::kPass] = 1;  // Make the dummy net always want to pass.

  MctsPlayer::Options options;
  options.random_seed = 17;
  auto player =
      absl::make_unique<TestablePlayer>(probs, 0, game_.get(), options);
  auto board = TestablePosition("");
  board.PlayMove("pass");
  player->InitializeGame(board);
  player->TreeSearch(1, std::numeric_limits<int>::max());
  EXPECT_EQ(0, CountPendingVirtualLosses(player->root()));
}

TEST_F(MctsPlayerTest, ExtractDataNormalEnd) {
  auto player =
      absl::make_unique<TestablePlayer>(game_.get(), MctsPlayer::Options());

  player->TreeSearch(1, std::numeric_limits<int>::max());
  player->PlayMove(Coord::kPass);
  player->TreeSearch(1, std::numeric_limits<int>::max());
  player->PlayMove(Coord::kPass);

  auto* root = player->root();
  EXPECT_TRUE(root->game_over());
  EXPECT_EQ(Color::kBlack, root->position.to_play());

  ASSERT_EQ(2, game_->num_moves());

  // White wins by komi
  EXPECT_EQ(-1, game_->result());
  EXPECT_EQ("W+7.5", game_->result_string());
}

TEST_F(MctsPlayerTest, ExtractDataResignEnd) {
  auto player =
      absl::make_unique<TestablePlayer>(game_.get(), MctsPlayer::Options());
  player->TreeSearch(1, std::numeric_limits<int>::max());
  player->PlayMove({0, 0});
  player->TreeSearch(1, std::numeric_limits<int>::max());
  player->PlayMove(Coord::kPass);
  player->TreeSearch(1, std::numeric_limits<int>::max());
  player->PlayMove(Coord::kResign);

  auto* root = player->root();

  // Black is winning on the board.
  EXPECT_LT(0, root->position.CalculateScore(game_->options().komi));
  EXPECT_EQ(-1, game_->result());
  EXPECT_EQ("W+R", game_->result_string());
}

TEST_F(MctsPlayerTest, UndoMove) {
  auto player =
      absl::make_unique<TestablePlayer>(game_.get(), MctsPlayer::Options());

  // Can't undo without first playing a move.
  EXPECT_FALSE(player->UndoMove());

  player->PlayMove(Coord::kPass);
  player->PlayMove(Coord::kPass);

  auto* root = player->root();
  EXPECT_TRUE(game_->game_over());
  EXPECT_EQ(Color::kBlack, root->position.to_play());
  ASSERT_EQ(2, game_->num_moves());
  EXPECT_EQ(-1, game_->result());
  EXPECT_EQ("W+7.5", game_->result_string());

  // Undo the last pass, the game should no longer be over.
  EXPECT_TRUE(player->UndoMove());

  root = player->root();
  EXPECT_FALSE(root->game_over());
  EXPECT_EQ(Coord::kPass, root->move);
  EXPECT_EQ(Color::kWhite, root->position.to_play());
  EXPECT_EQ(1, game_->num_moves());
}

// Soft pick won't work correctly if none of the points on the board have been
// visited (for example, if a model puts all its reads into pass). This is the
// only case where soft pick should return kPass.
TEST_F(MctsPlayerTest, SoftPickWithNoVisits) {
  Random rnd(25323, 1);
  auto player =
      absl::make_unique<TestablePlayer>(game_.get(), MctsPlayer::Options());
  EXPECT_EQ(Coord::kPass, player->mutable_tree()->PickMove(&rnd, true));
}

}  // namespace
}  // namespace minigo

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::minigo::zobrist::Init(614944751);
  return RUN_ALL_TESTS();
}
