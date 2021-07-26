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

#ifndef LIBCORAL_CORAL_LEARN_BACKPROP_TEST_UTILS_H_
#define LIBCORAL_CORAL_LEARN_BACKPROP_TEST_UTILS_H_

#include <functional>

#include "Eigen/Core"
#include "coral/learn/backprop/softmax_regression_model.h"

namespace coral {

// Func takes in tensor and outputs a scalar value
using Func = std::function<float(const Eigen::MatrixXf&)>;

// Gets numerical gradient of f at point x to use in backprop gradient checking
// uses dx = f(x+h)-f(x-h)/(2*h) as numerical approximation where h is epsilon
Eigen::MatrixXf NumericalGradient(Func f, Eigen::MatrixXf* x,
                                  float epsilon = 1e-3);

// Helper function to generate data in which examples from the same class are
// drawn from the same MultiVariate Normal (MVN) Distribution.
// Note that this function generates real random values. If this leads to
// flakiness consider change it to create same pseudo random value sequence.
TrainingData GenerateMvnRandomData(const std::vector<int>& class_sizes,
                                   const std::vector<Eigen::VectorXf>& means,
                                   const std::vector<Eigen::MatrixXf>& cov_mats,
                                   int num_train);

// Helper function to generate data in which examples are drawn from Uniform
// Distribution. Note that this function generates real random values. If this
// leads to flakiness consider change it to create same pseudo random value
// sequence.
TrainingData GenerateUniformRandomData(const std::vector<int>& class_sizes,
                                       int feature_dim, int num_train);

}  // namespace coral
#endif  // LIBCORAL_CORAL_LEARN_BACKPROP_TEST_UTILS_H_
