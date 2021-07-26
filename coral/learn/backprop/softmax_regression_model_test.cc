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

#include <algorithm>
#include <numeric>
#include <vector>

#include "coral/learn/backprop/test_utils.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;
using ::testing::Test;

TEST(SoftmaxRegressionModelTest, SeparableData) {
  int num_train = 200;
  int num_val = 30;
  int num_classes = 3;
  std::vector<int> class_sizes;
  int inputs_per_class = static_cast<int>((num_train + num_val) / num_classes);

  // Distribute data evenly among different classes.
  class_sizes = std::vector<int>(num_classes, inputs_per_class);
  class_sizes[0] =
      (num_train + num_val) -
      std::accumulate(class_sizes.begin(), class_sizes.end() - 1, 0);

  // 3 is chosen, such that each pair of mean is over 6 "sigma" distance apart.
  // This makes it harder for the classes to "touch" each other.
  // Reference: https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
  VectorXf mean_one(2), mean_two(2), mean_three(2);
  mean_one(0) = 1 * 3;
  mean_one(1) = 1 * 3;
  mean_two(0) = -1 * 3;
  mean_two(1) = -1 * 3;
  mean_three(0) = 1 * 3;
  mean_three(1) = -1 * 3;
  std::vector<VectorXf> means = {mean_one, mean_two, mean_three};
  int feature_dim = mean_one.size();
  MatrixXf cov_mat = MatrixXf::Identity(feature_dim, feature_dim);
  std::vector<MatrixXf> cov_mats = {cov_mat, cov_mat, cov_mat};

  SoftmaxRegressionModel model(feature_dim, num_classes);
  TrainingData my_data =
      GenerateMvnRandomData(class_sizes, means, cov_mats, num_train);
  model.Train(my_data, /*train_config=*/{5, 100, 1}, /*learning_rate=*/.01);

  EXPECT_GT(model.GetAccuracy(my_data.training_data, my_data.training_labels),
            0.94);
}

TEST(SoftmaxRegressionModelTest, NonSeperableData) {
  int num_train = 200;
  int num_val = 30;
  int num_classes = 3;
  std::vector<int> class_sizes;
  int inputs_per_class = static_cast<int>((num_train + num_val) / num_classes);

  // Distribute data evenly among different classes.
  class_sizes = std::vector<int>(num_classes, inputs_per_class);
  class_sizes[0] =
      (num_train + num_val) -
      std::accumulate(class_sizes.begin(), class_sizes.end() - 1, 0);

  VectorXf mean_one(2), mean_two(2), mean_three(2);
  mean_one(0) = 1;
  mean_one(1) = 1;
  mean_two(0) = -1;
  mean_two(1) = -1;
  mean_three(0) = 1;
  mean_three(1) = -1;
  std::vector<VectorXf> means = {mean_one, mean_two, mean_three};
  int feature_dim = mean_one.size();
  MatrixXf cov_mat = MatrixXf::Identity(feature_dim, feature_dim);
  std::vector<MatrixXf> cov_mats = {cov_mat, cov_mat, cov_mat};

  SoftmaxRegressionModel model(feature_dim, num_classes);
  TrainingData my_data =
      GenerateMvnRandomData(class_sizes, means, cov_mats, num_train);

  model.Train(my_data, /*train_config=*/{10, 100, 1}, /*learning_rate=*/.1);
  EXPECT_GT(model.GetAccuracy(my_data.training_data, my_data.training_labels),
            0.68);
}
}  // namespace
}  // namespace coral
