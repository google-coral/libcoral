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

#ifndef LIBCORAL_CORAL_ERROR_REPORTER_H_
#define LIBCORAL_CORAL_ERROR_REPORTER_H_

#include <sstream>
#include <string>

#include "tensorflow/lite/stateful_error_reporter.h"

namespace coral {

class EdgeTpuErrorReporter : public tflite::StatefulErrorReporter {
 public:
  // We declared two functions with name 'Report', so the variadic Report
  // function in tflite::ErrorReporter is hidden.
  // See https://isocpp.org/wiki/faq/strange-inheritance#hiding-rule.
  using tflite::ErrorReporter::Report;

  int Report(const char* format, va_list args) override;

  std::string message() override;

 private:
  static constexpr int kBufferSize = 1024;
  char last_message_[kBufferSize];
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_ERROR_REPORTER_H_
