// Copyright 2019 Google LLC
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

#include "cc/async/thread.h"

#include <algorithm>

#include "cc/logging.h"

namespace minigo {

Thread::Thread(std::string name) : name_(std::move(name)) {
  constexpr size_t kMaxLen = 15;
  if (name_.size() > kMaxLen) {
    name_.resize(kMaxLen);
  }
}

Thread::~Thread() = default;

void Thread::Start() {
  impl_ = std::thread([this] { Run(); });
  if (!name_.empty()) {
    pthread_setname_np(handle(), name_.c_str());
  }
}

void Thread::Join() {
  MG_CHECK(impl_.joinable());
  impl_.join();
}

void LambdaThread::Run() { closure_(); }

}  // namespace minigo
