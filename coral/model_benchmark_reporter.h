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

#ifndef LIBCORAL_CORAL_MODEL_BENCHMARK_REPORTER_H_
#define LIBCORAL_CORAL_MODEL_BENCHMARK_REPORTER_H_

#include "benchmark/benchmark.h"

namespace coral {

// Customized benchmark result reporter. Output looks like:
/*
models = [
    {
      'filename': 'model1_edgetpu.tflite',
      'latency': '10 ms',
    },
    {
      'filename': 'model2.tflite',
      'latency': '30 ms',
    },
  ]
*/
class ModelBenchmarkReporter : public benchmark::BenchmarkReporter {
 public:
  ModelBenchmarkReporter() {}
  virtual bool ReportContext(const Context& context);
  virtual void ReportRuns(const std::vector<Run>& reports);
  virtual void Finalize();

 private:
  void PrintRunData(const Run& run);
};

}  // namespace coral

#endif  // LIBCORAL_CORAL_MODEL_BENCHMARK_REPORTER_H_
