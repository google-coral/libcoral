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

#include "coral/model_benchmark_reporter.h"

#include <iomanip>
#include <ostream>

namespace coral {

bool ModelBenchmarkReporter::ReportContext(const Context& context) {
  GetOutputStream() << "models_benchmark = [" << std::endl;
  return true;
}

void ModelBenchmarkReporter::ReportRuns(const std::vector<Run>& reports) {
  // print results for each run
  for (const auto& run : reports) {
    PrintRunData(run);
  }
}

void ModelBenchmarkReporter::PrintRunData(const Run& run) {
  std::ostream& Out = GetOutputStream();
  Out << "    {" << std::endl;
  Out << "      'filename': '" << run.report_label << "'," << std::endl;
  Out << "      'latency': '" << std::fixed << std::setprecision(1)
      << run.GetAdjustedRealTime() / 1000000.0 << " ms'," << std::endl;
  Out << "    }," << std::endl;
}

void ModelBenchmarkReporter::Finalize() {
  GetOutputStream() << "  ]" << std::endl;
}

}  // namespace coral
