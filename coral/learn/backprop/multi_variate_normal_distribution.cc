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

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>

#include "glog/logging.h"

static std::default_random_engine generator(
    std::chrono::system_clock::now().time_since_epoch().count());

namespace coral {
using Eigen::MatrixXf;
using Eigen::VectorXf;

MultiVariateNormalDistribution::MultiVariateNormalDistribution(
    const VectorXf& mean, const MatrixXf& cov)
    : mean_(mean), cov_(cov), dim_(cov.rows()) {
  solver_.compute(cov_, true);
  auto eigen_values = solver_.eigenvalues().real();
  VLOG(1) << "eigen_values is " << eigen_values;
  VLOG(1) << "eigen_vector is " << solver_.eigenvectors().real();
  MatrixXf eigen_vectors = solver_.eigenvectors().real();
  VLOG(1) << "eigen_vector is " << solver_.eigenvectors().real();
  MatrixXf Q = eigen_vectors;
  for (int i = 0; i < eigen_vectors.cols(); i++) {
    float norm = Q.col(i).squaredNorm();
    Q.col(i) /= norm;
  }
  VLOG(1) << "Q is " << Q;
  VectorXf sqrt_lambda(eigen_values.size());
  for (int i = 0; i < eigen_values.size(); i++) {
    sqrt_lambda(i) = std::sqrt(static_cast<float>(eigen_values(i)));
  }
  MatrixXf sqrt_lambda_matrix = sqrt_lambda.asDiagonal();
  VLOG(1) << "sqrt_lambda_matrix is " << sqrt_lambda_matrix;
  p_ = Q * sqrt_lambda_matrix;
  VLOG(1) << "P is " << p_;
}

MatrixXf MultiVariateNormalDistribution::Sample(int num) {
  // Initialize x;
  MatrixXf x(dim_, num);
  for (int i = 0; i < dim_; i++) {
    for (int j = 0; j < num; j++) {
      x(i, j) = rand_gaussian_(generator);
    }
  }
  MatrixXf ret = p_ * x;
  ret.colwise() += mean_;
  return ret;
}

}  // namespace coral
