
#include "coral/error_reporter.h"

#include "absl/memory/memory.h"
#include "gtest/gtest.h"

namespace coral {
TEST(ErrorReporterTest, CheckReport) {
  std::unique_ptr<tflite::StatefulErrorReporter> reporter =
      absl::make_unique<EdgeTpuErrorReporter>();
  reporter->Report("test %d", 1);
  EXPECT_EQ("test 1", reporter->message());

  reporter->Report("test %d %s", 2, "test");
  EXPECT_EQ("test 2 test", reporter->message());
}

TEST(ErrorReporterTest, CheckEmptyMessage) {
  std::unique_ptr<tflite::StatefulErrorReporter> reporter =
      absl::make_unique<EdgeTpuErrorReporter>();
  EXPECT_EQ("", reporter->message());
}

}  // namespace coral
