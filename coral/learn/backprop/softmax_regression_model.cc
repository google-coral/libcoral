/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "coral/learn/backprop/softmax_regression_model.h"

#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "coral/learn/backprop/multi_variate_normal_distribution.h"
#include "coral/learn/utils.h"
#include "glog/logging.h"

namespace coral {
using ::Eigen::MatrixXf;

SoftmaxRegressionModel::SoftmaxRegressionModel(int feature_dim, int num_classes,
                                               float weight_scale, float reg)
    : feature_dim_(feature_dim),
      num_classes_(num_classes),
      weight_scale_(weight_scale),
      reg_(reg) {
  // Randomly set weights for mat_w_ from a gaussian distribution.
  mat_w_ = MatrixXf::Ones(feature_dim_, num_classes_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dist(0, 1);
  auto random = [&gen, &dist](float x) -> float { return dist(gen); };
  mat_w_ = weight_scale_ * mat_w_.unaryExpr(random);

  // Set weights for vec_b_ to zero.
  vec_b_ = MatrixXf::Zero(1, num_classes_);

  caches_ = std::vector<MatrixXf>(3);
  logit_min_ = std::numeric_limits<float>::infinity();
  logit_max_ = -std::numeric_limits<float>::infinity();

  VLOG(1) << "DONE INITIALIZING";
}

float SoftmaxRegressionModel::GetLoss(const MatrixXf& mat_x,
                                      const MatrixXf& labels) {
  // cache[0] is tensor logits
  caches_[0] = FullyConnected(mat_x, mat_w_, vec_b_);
  logit_min_ = std::min(logit_min_, caches_[0].minCoeff());
  logit_max_ = std::max(logit_max_, caches_[0].maxCoeff());
  // cache[1] is tensor probs
  caches_[1] = Softmax(caches_[0]);
  // cache[2] is tensor loss
  caches_[2] = CrossEntropyLoss(labels, caches_[1]);
  // add regularization term
  VLOG(1) << "Adding regularization term to loss";
  return caches_[2](0, 0) +
         0.5 * reg_ * (mat_w_.transpose() * mat_w_).array().sum();
}

std::vector<MatrixXf> SoftmaxRegressionModel::GetGrads(const MatrixXf& mat_x,
                                                       const MatrixXf& labels) {
  MatrixXf dprobs = CrossEntropyGradient(labels, caches_[1]);
  MatrixXf dlogits = SoftmaxGradient(caches_[0], dprobs);
  std::vector<MatrixXf> xwb_grads =
      FullyConnectedGradient(mat_x, mat_w_, vec_b_, dlogits);
  return {xwb_grads[1], xwb_grads[2]};
}

// Helper function to compute the argmax for each row of input tensor.
// Eigen does not have a built-in rowwise operation for argmax.
std::vector<int> RowwiseArgmax(const MatrixXf& input) {
  std::vector<int> output(input.rows(), -1);
  MatrixXf::Index argmax;
  for (int i = 0; i < input.rows(); i++) {
    input.row(i).maxCoeff(&argmax);
    output[i] = argmax;
  }
  return output;
}

// Helper function to get a random set of b indices for the batch where b is the
// batch_size.
// The indices are in the range (0, n) where n+1 is number of rows possible.
std::vector<int> GetBatchIndices(const MatrixXf& tensor, int batch_size) {
  std::random_device random_device;
  std::mt19937 mersenne_engine{random_device()};
  std::uniform_int_distribution<int> dist{0,
                                          static_cast<int>(tensor.rows() - 1)};
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
  std::vector<int> vec(batch_size);
  std::generate(std::begin(vec), std::end(vec), gen);
  return vec;
}

std::vector<int> SoftmaxRegressionModel::RunInference(const MatrixXf& mat_x) {
  return RowwiseArgmax(FullyConnected(mat_x, mat_w_, vec_b_));
}

float SoftmaxRegressionModel::GetAccuracy(const MatrixXf& mat_x,
                                          const std::vector<int>& labels) {
  CHECK_EQ(labels.size(), mat_x.rows());
  const auto result = RunInference(mat_x);
  int correct = 0;
  for (int r = 0; r < result.size(); ++r) {
    if (result[r] == labels[r]) ++correct;
  }
  return static_cast<float>(correct) / labels.size();
}

void SoftmaxRegressionModel::Train(const TrainingData& data,
                                   const TrainConfig& train_config,
                                   float learning_rate) {
  // For each iteration in num_iter, use a random batch of inputs of size
  // batch_size to calculate gradients and learn model weights.
  for (int i = 0; i < train_config.num_iter; i++) {
    const auto& batch_indices =
        GetBatchIndices(data.training_data, train_config.batch_size);
    MatrixXf train_batch, labels_batch;
    train_batch = data.training_data(batch_indices, Eigen::indexing::all);

    // Create one-hot label vectors
    labels_batch = MatrixXf::Zero(train_config.batch_size, num_classes_);
    for (int r = 0; r < train_config.batch_size; ++r) {
      labels_batch(r, data.training_labels[batch_indices[r]]) = 1.0f;
    }
    float loss = GetLoss(train_batch, labels_batch);

    SgdUpdate(GetGrads(train_batch, labels_batch), learning_rate,
              {&mat_w_, &vec_b_});

    if (train_config.print_every > 0 && i % train_config.print_every == 0) {
      LOG(INFO) << "Loss: " << loss;
      LOG(INFO) << "Train Acc: "
                << GetAccuracy(data.training_data, data.training_labels);
      LOG(INFO) << "Valid Acc: "
                << GetAccuracy(data.validation_data, data.validation_labels);
    }
  }
}

void SoftmaxRegressionModel::AppendLayersToEmbeddingExtractor(
    const tflite::Model& embedding_extractor_model,
    flatbuffers::FlatBufferBuilder* fbb) const {
  LOG(INFO) << "Logit min: " << logit_min_ << ", max: " << logit_max_;
  CHECK_EQ(AppendFullyConnectedAndSoftmaxLayerToModel(embedding_extractor_model,
                                                      fbb, mat_w_, vec_b_,
                                                      logit_min_, logit_max_),
           absl::OkStatus());
}

}  // namespace coral
