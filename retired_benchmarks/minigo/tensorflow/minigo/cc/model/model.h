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

#ifndef CC_MODEL_MODEL_H_
#define CC_MODEL_MODEL_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "cc/color.h"
#include "cc/constants.h"
#include "cc/inline_vector.h"
#include "cc/model/features.h"
#include "cc/model/types.h"
#include "cc/position.h"
#include "cc/symmetries.h"

namespace minigo {

// TODO(tommadams): replace all std::vector parameters with absl::Span.
class Model {
 public:
  // Fills a batch of inference outputs from policy and value tensors.
  // Args:
  //   model_inputs: the same inputs that were passed to `SetFeatures`.
  //   policy: the policy output from the model.
  //   value: the value output from the model.
  //   model_outputs: the model outputs to fill. `model_inputs.size()` must ==
  //                  `model_outputs.size()`.
  // Models that produce quantized outputs should unquantize them into
  // `Tensor<float>` objects before calling GetOutputs.
  static void GetOutputs(absl::Span<const ModelInput* const> inputs,
                         const Tensor<float>& policy,
                         const Tensor<float>& value,
                         absl::Span<ModelOutput*> outputs);

  static void ApplySymmetry(symmetry::Symmetry sym, const ModelOutput& src,
                            ModelOutput* dst);

  Model(std::string name, const FeatureDescriptor& feature_desc);
  virtual ~Model();

  const std::string& name() const { return name_; }
  const FeatureDescriptor& feature_descriptor() const { return feature_desc_; }

  // TODO(tommadams): remove the model_name out parameter: it's no longer needed
  // with the new concurrent_selfplay implementation.
  virtual void RunMany(const std::vector<const ModelInput*>& inputs,
                       std::vector<ModelOutput*>* outputs,
                       std::string* model_name) = 0;

 private:
  const std::string name_;
  const FeatureDescriptor feature_desc_;
};

}  // namespace minigo

#endif  //  CC_MODEL_MODEL_H_
