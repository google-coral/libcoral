#include "coral/error_reporter.h"

#include <cstring>

namespace coral {

int EdgeTpuErrorReporter::Report(const char* format, va_list args) {
  std::memset(last_message_, 0, kBufferSize);
  return vsnprintf(last_message_, kBufferSize, format, args);
}

std::string EdgeTpuErrorReporter::message() { return last_message_; }

}  // namespace coral
