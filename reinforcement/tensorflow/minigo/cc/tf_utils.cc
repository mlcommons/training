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

#include "cc/tf_utils.h"

#include <algorithm>
#include <array>
#include <memory>

#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/model/model.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

using tensorflow::io::RecordWriter;
using tensorflow::io::RecordWriterOptions;

namespace minigo {
namespace tf_utils {

namespace {

tensorflow::Feature MakeBytesFeature(const Tensor<uint8_t>& src) {
  int size = src.shape.num_elements();
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(src.data), size);
  return feature;
}

template <size_t N>
tensorflow::Feature MakeBytesFeature(const std::array<float, N>& data) {
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(data.data()), sizeof(float) * data.size());
  return feature;
}

// Converts board features, and the pi & value outputs of MTCS to a tensorflow
// example proto.
tensorflow::Example MakeTfExample(const Tensor<uint8_t>& features,
                                  const std::array<float, kNumMoves>& pi,
                                  float Q, int N, Coord c, float outcome) {
  tensorflow::Example example;
  auto& dst_features = *example.mutable_features()->mutable_feature();

  // The input features are expected to be uint8 bytes.
  dst_features["x"] = MakeBytesFeature(features);

  // pi is expected to be a float array serialized as bytes.
  dst_features["pi"] = MakeBytesFeature(pi);

  // outcome is a single float.
  dst_features["outcome"].mutable_float_list()->add_value(outcome);

  // Q is a single float.
  dst_features["q"].mutable_float_list()->add_value(Q);

  // Number of reads is a single int.
  dst_features["n"].mutable_int64_list()->add_value(N);

  // The move played is a single int.
  dst_features["c"].mutable_int64_list()->add_value(c);

  return example;
}

// Writes a list of tensorflow Example protos to a zlib compressed TFRecord
// file.
void WriteTfExamples(const std::string& path,
                     const std::vector<tensorflow::Example>& examples) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  options.zlib_options.compression_level = 2;
  RecordWriter writer(file.get(), options);

  std::string data;
  for (const auto& example : examples) {
    example.SerializeToString(&data);
    TF_CHECK_OK(writer.WriteRecord(data));
  }

  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}

}  // namespace

std::vector<tensorflow::Example> MakeExamples(
    const FeatureDescriptor& feature_desc, const Game& game) {
  // Write the TensorFlow examples.
  std::vector<tensorflow::Example> examples;
  examples.reserve(game.num_moves());

  auto shape = feature_desc.GetInputShape(1);
  BoardFeatureBuffer<uint8_t> features_buffer;
  Tensor<uint8_t> features(shape, features_buffer.data());

  for (size_t i = 0; i < game.moves().size(); ++i) {
    const auto* move = game.moves()[i].get();
    if (!move->is_trainable()) {
      continue;
    }

    ModelInput input;
    input.sym = symmetry::kIdentity;
    game.GetPositionHistory(i, kMaxPositionHistory, &input.position_history);

    feature_desc.set_bytes({&input}, &features);
    examples.push_back(MakeTfExample(features, move->search_pi.value(), move->Q,
                                     move->N, move->c, game.result()));
  }
  return examples;
}

void WriteGameExamples(const std::string& output_dir,
                       const std::string& output_name,
                       const FeatureDescriptor& feature_desc,
                       const Game& game) {
  MG_CHECK(file::RecursivelyCreateDir(output_dir));
  auto output_path = file::JoinPath(output_dir, output_name + ".tfrecord.zz");

  auto examples = MakeExamples(feature_desc, game);
  WriteTfExamples(output_path, examples);
}

}  // namespace tf_utils
}  // namespace minigo
