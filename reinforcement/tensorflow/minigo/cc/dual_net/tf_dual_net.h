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

#ifndef CC_DUAL_NET_TF_DUAL_NET_H_
#define CC_DUAL_NET_TF_DUAL_NET_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "cc/model/model.h"
#include "cc/model/factory.h"

namespace minigo {

class TfDualNetFactory : public ModelFactory {
 public:
  explicit TfDualNetFactory(absl::string_view device);

  std::unique_ptr<Model> NewModel(const ModelDefinition& def) override;

 private:
  bool place_on_gpu_ = true;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_TF_DUAL_NET_H_
