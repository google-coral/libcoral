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

#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "benchmark/benchmark.h"
#include "coral/model_benchmark_reporter.h"

ABSL_FLAG(bool, use_custom_out_format, false,
          "Whether use the custom output format to save benchmark results.");

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  absl::ParseCommandLine(argc, argv);

  std::unique_ptr<benchmark::BenchmarkReporter> file_reporter;
  if (absl::GetFlag(FLAGS_use_custom_out_format)) {
    file_reporter.reset(new coral::ModelBenchmarkReporter());
  }

  benchmark::RunSpecifiedBenchmarks(nullptr, file_reporter.get());
  return 0;
}
