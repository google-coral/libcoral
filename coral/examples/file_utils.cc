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

#include "coral/examples/file_utils.h"

#include <cstdio>
#include <fstream>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "glog/logging.h"

namespace coral {

std::unordered_map<int, std::string> ReadLabelFile(
    const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  CHECK(file) << "Cannot open " << file_path;

  std::unordered_map<int, std::string> labels;
  int counter = 0;
  std::string line;
  while (std::getline(file, line)) {
    absl::RemoveExtraAsciiWhitespace(&line);
    std::vector<std::string> fields =
        absl::StrSplit(line, absl::MaxSplits(' ', 1));
    int id;
    if (fields.size() == 2 && absl::SimpleAtoi(fields[0], &id)) {
      labels.insert({id, fields[1]});
    } else {
      labels.insert({counter, line});
    }
    ++counter;
  }
  return labels;
}

void ReadFileToOrDie(const std::string& file_path, char* data, size_t size) {
  std::ifstream file(file_path, std::ios::binary);
  CHECK(file) << "Cannot open " << file_path;
  CHECK(file.read(data, size))
      << "Cannot read " << size << " bytes from " << file_path;
  CHECK_EQ(file.peek(), EOF)
      << file_path << " size must match input size of " << size << " bytes";
}

}  // namespace coral
