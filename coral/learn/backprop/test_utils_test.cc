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

#include "Eigen/Core"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

using ::Eigen::MatrixXf;
using ::Eigen::VectorXf;
using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(TestUtilsTest, NumericalGradientOfXtimesXtranspose) {
  MatrixXf mat_x(1, 4);
  mat_x << 1, 2, 3, 4;
  auto x_squared = [](const MatrixXf& x) { return (x * x.transpose())(0, 0); };
  MatrixXf dx = NumericalGradient(x_squared, &mat_x);
  MatrixXf mat_y_expected = 2 * mat_x;
  EXPECT_THAT(dx.reshaped(),
              Pointwise(FloatNear(1e-3), mat_y_expected.reshaped()));
}

TEST(TestUtilsTest, GenerateFakeData_NormalDistribution) {
  constexpr int kNumClasses = 2;
  constexpr int kTotalNumSamples = 300;
  constexpr int kNumTraining = 200;
  constexpr int kFeatureDim = 7;
  std::vector<VectorXf> means;
  std::vector<MatrixXf> cov_mats;
  means.reserve(kNumClasses);
  cov_mats.reserve(kNumClasses);
  for (int i = 0; i < kNumClasses; ++i) {
    means.push_back(VectorXf::Random(kFeatureDim));
    cov_mats.push_back(MatrixXf::Random(kFeatureDim, kFeatureDim));
  }
  const std::vector<int> class_sizes(kNumClasses,
                                     kTotalNumSamples / kNumClasses);
  const auto fake_data =
      GenerateMvnRandomData(class_sizes, means, cov_mats, kNumTraining);
  EXPECT_EQ(fake_data.training_data.rows(), kNumTraining);
  EXPECT_EQ(fake_data.training_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.validation_data.rows(), kTotalNumSamples - kNumTraining);
  EXPECT_EQ(fake_data.validation_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.training_labels.size(), kNumTraining);
  EXPECT_EQ(fake_data.validation_labels.size(),
            kTotalNumSamples - kNumTraining);
}

TEST(TestUtilsTest, GenerateFakeData_UniformDistribution) {
  constexpr int kNumClasses = 2;
  constexpr int kTotalNumSamples = 300;
  constexpr int kNumTraining = 200;
  constexpr int kFeatureDim = 7;
  const std::vector<int> class_sizes(kNumClasses,
                                     kTotalNumSamples / kNumClasses);
  const auto fake_data =
      GenerateUniformRandomData(class_sizes, kFeatureDim, kNumTraining);
  EXPECT_EQ(fake_data.training_data.rows(), kNumTraining);
  EXPECT_EQ(fake_data.training_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.validation_data.rows(), kTotalNumSamples - kNumTraining);
  EXPECT_EQ(fake_data.validation_data.cols(), kFeatureDim);
  EXPECT_EQ(fake_data.training_labels.size(), kNumTraining);
  EXPECT_EQ(fake_data.validation_labels.size(),
            kTotalNumSamples - kNumTraining);
}

}  // namespace
}  // namespace coral
