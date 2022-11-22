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

#ifndef CC_CONSTANTS_H_
#define CC_CONSTANTS_H_

namespace minigo {

// MINIGO_BOARD_SIZE is defined differently by different build targets:
//  - minigo_9 defines MINIGO_BOARD_SIZE as 9.
//  - minigo_19 defines MINIGO_BOARD_SIZE as 19.
// When building through Bazel, MINIGO_BOARD_SIZE is automatically defined for
// all build targets that depend on either minigo_9 or minigo_19.
constexpr int kN = MINIGO_BOARD_SIZE;

// kN * kN possible points on the board.
constexpr int kNumPoints = kN * kN;

// points on the board, plus pass.
constexpr int kNumMoves = kNumPoints + 1;

// Safe lower bound of the number of moves to reach whole-board pass-alive
// state. We use this number to skip calculating whether the whole board is
// pass-alive for the early part of the game:
//   133 for 19x19, 27 for 9x9
// The best positions we've found for these are:
//   141 for 19x19, 40 for 9x9
// but there may be more optimal sequences of moves.
constexpr int kMinPassAliveMoves = kN * ((kN + 2) / 3);

constexpr float kDefaultKomi = 7.5;

constexpr float kDirichletAlpha = 0.03f * 361 / kNumPoints;

// static constexpr float kPuct = 2.50;
static constexpr float kUct_base = 19652;
static constexpr float kUct_init = 1.25;

// Maximum number of moves we look back in history when preparing input for an
// inference.
static constexpr int kMaxPositionHistory = 8;

}  // namespace minigo

#endif  //  CC_CONSTANTS_H_
