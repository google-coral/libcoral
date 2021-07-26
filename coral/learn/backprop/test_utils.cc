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

#include "coral/learn/backprop/test_utils.h"

#include <algorithm>
#include <numeric>

#include "coral/learn/backprop/multi_variate_normal_distribution.h"
#include "glog/logging.h"

namespace coral {

namespace {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;

// Helper function that shuffles data and split into training and validation
// subsets.
TrainingData ShuffleAndSplitData(const MatrixXf& data_matrix,
                                 const std::vector<int> labels_vector,
                                 int num_train) {
  const int total_rows = data_matrix.rows();
  std::vector<int> shuffled_indices(total_rows, -1);
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
  std::random_device rd;
  std::shuffle(shuffled_indices.begin(), shuffled_indices.end(),
               std::mt19937(rd()));
  MatrixXf shuffled_data =
      MatrixXf::Zero(data_matrix.rows(), data_matrix.cols());
  shuffled_data = data_matrix(shuffled_indices, Eigen::all);
  std::vector<int> shuffled_labels(total_rows, -1);
  for (int i = 0; i < total_rows; ++i) {
    shuffled_labels[i] = labels_vector[shuffled_indices[i]];
  }

  // Eigen::seq boundaries are inclusive on both sides.
  TrainingData fake_data;
  fake_data.training_data =
      shuffled_data(Eigen::seq(0, num_train - 1), Eigen::all);
  fake_data.validation_data =
      shuffled_data(Eigen::seq(num_train, Eigen::last), Eigen::all);

  fake_data.training_labels.assign(shuffled_labels.begin(),
                                   shuffled_labels.begin() + num_train);
  fake_data.validation_labels.assign(shuffled_labels.begin() + num_train,
                                     shuffled_labels.end());

  CHECK_EQ(fake_data.training_data.rows(), fake_data.training_labels.size());
  CHECK_EQ(fake_data.validation_data.rows(),
           fake_data.validation_labels.size());
  return fake_data;
}
}  // namespace

MatrixXf NumericalGradient(Func f, MatrixXf* x, float epsilon) {
  MatrixXf dx = Eigen::MatrixXf::Zero(x->rows(), x->cols());
  float b, a;
  for (int i = 0; i < x->rows(); i++) {
    for (int j = 0; j < x->cols(); j++) {
      const float val = (*x)(i, j);

      (*x)(i, j) = val + epsilon;
      b = f(*x);

      (*x)(i, j) = val - epsilon;
      a = f(*x);
      (*x)(i, j) = val;

      dx(i, j) = (b - a) / (2 * epsilon);
    }
  }
  return dx;
}

TrainingData GenerateMvnRandomData(const std::vector<int>& class_sizes,
                                   const std::vector<VectorXf>& means,
                                   const std::vector<MatrixXf>& cov_mats,
                                   int num_train) {
  CHECK_EQ(class_sizes.size(), means.size());
  CHECK_EQ(class_sizes.size(), cov_mats.size());

  TrainingData fake_data;
  int total_rows = std::accumulate(class_sizes.begin(), class_sizes.end(), 0);
  int total_cols = cov_mats[0].rows();
  MatrixXf data_matrix = MatrixXf::Zero(total_rows, total_cols);
  std::vector<int> labels_vector;
  labels_vector.reserve(total_rows);
  int start_index = 0;
  for (int i = 0; i < class_sizes.size(); i++) {
    int n = class_sizes[i];
    MultiVariateNormalDistribution dist(means[i], cov_mats[i]);
    MatrixXf samples = dist.Sample(n);
    // Eigen::seq boundaries are inclusive on both sides.
    data_matrix(Eigen::seq(start_index, start_index + n - 1), Eigen::all) =
        samples.transpose();
    labels_vector.insert(labels_vector.end(), n, i);
    start_index += n;
  }

  return ShuffleAndSplitData(data_matrix, labels_vector, num_train);
}

TrainingData GenerateUniformRandomData(const std::vector<int>& class_sizes,
                                       int feature_dim, int num_train) {
  TrainingData fake_data;
  int total_rows = std::accumulate(class_sizes.begin(), class_sizes.end(), 0);
  int total_cols = feature_dim;
  MatrixXf data_matrix = MatrixXf::Zero(total_rows, total_cols);
  std::vector<int> labels_vector;
  labels_vector.reserve(total_rows);
  int start_index = 0;
  for (int i = 0; i < class_sizes.size(); i++) {
    int n = class_sizes[i];
    MatrixXf samples = MatrixXf::Random(total_cols, n);
    // Eigen::seq boundaries are inclusive on both sides.
    data_matrix(Eigen::seq(start_index, start_index + n - 1), Eigen::all) =
        samples.transpose();
    labels_vector.insert(labels_vector.end(), n, i);
    start_index += n;
  }

  return ShuffleAndSplitData(data_matrix, labels_vector, num_train);
}
}  // namespace coral
