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

#ifndef LIBCORAL_CORAL_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#define LIBCORAL_CORAL_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
#include <random>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace coral {

// Multi variate normal distribution implemented with Eigen library.
class MultiVariateNormalDistribution {
 public:
  MultiVariateNormalDistribution(const Eigen::VectorXf& mean,
                                 const Eigen::MatrixXf& cov);

  // Samples 'num' samples from distribution.
  // Returns a [dim, num] shape matrix.
  Eigen::MatrixXf Sample(int num);

 private:
  // Mean of the distribution.
  Eigen::VectorXf mean_;

  // Covariance matrix of the distribution.
  Eigen::MatrixXf cov_;

  // Multiplies this matrix with a random variable X which is drawn from
  // N(0, I) will produce a sample drawn from N(0, cov_).
  Eigen::MatrixXf p_;

  // The dimension of the covariance matrix.
  int dim_;

  // Eigen solver which is used to compute eigen value and eigen vectors.
  Eigen::EigenSolver<Eigen::MatrixXf> solver_;

  // Gaussian random number generator.
  std::normal_distribution<float> rand_gaussian_;
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_LEARN_BACKPROP_MULTI_VARIATE_NORMAL_DISTRIBUTION_H_
