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

#include "coral/learn/backprop/multi_variate_normal_distribution.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace coral {
using Eigen::MatrixXf;
using Eigen::VectorXf;

static MatrixXf Covariance(const MatrixXf& mat) {
  MatrixXf centered = mat.rowwise() - mat.colwise().mean();
  return (centered.adjoint() * centered) / static_cast<float>((mat.rows() - 1));
}

TEST(MultiVariateNormalDistributionTest, Test) {
  VectorXf mean(2);
  mean << 2.0, 3.0;
  MatrixXf cov(2, 2);
  cov << 1, 0.3, 0.3, 0.6;
  VLOG(1) << cov;
  MultiVariateNormalDistribution dist(mean, cov);
  auto samples = dist.Sample(10000);
  auto samples_means = samples.rowwise().mean();
  EXPECT_NEAR(2.0, samples_means[0], 0.1);
  EXPECT_NEAR(3.0, samples_means[1], 0.1);
  auto samples_cov = Covariance(samples.transpose());
  VLOG(1) << "samples cov is " << samples_cov;
  EXPECT_NEAR(1.0, samples_cov(0, 0), 0.1);
  EXPECT_NEAR(0.3, samples_cov(0, 1), 0.1);
  EXPECT_NEAR(0.3, samples_cov(1, 0), 0.1);
  EXPECT_NEAR(0.6, samples_cov(1, 1), 0.1);
}

}  // namespace coral
