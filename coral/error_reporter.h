#ifndef EDGETPU_CPP_ERROR_REPORTER_H_
#define EDGETPU_CPP_ERROR_REPORTER_H_

#include <sstream>
#include <string>

#include "tensorflow/lite/stateful_error_reporter.h"

namespace coral {

class EdgeTpuErrorReporter : public tflite::StatefulErrorReporter {
 public:
  // We declared two functions with name 'Report', so that the variadic Report
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

#endif  // EDGETPU_CPP_ERROR_REPORTER_H_
