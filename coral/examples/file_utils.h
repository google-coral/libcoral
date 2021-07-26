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

#ifndef LIBCORAL_CORAL_EXAMPLES_FILE_UTILS_H_
#define LIBCORAL_CORAL_EXAMPLES_FILE_UTILS_H_

#include <cstddef>
#include <string>
#include <unordered_map>

namespace coral {
// Reads labels from text file and store it in an unordered_map.
//
// This function supports the following format:
//   Each line contains id and description separated by a space.
//   Example: '0 cat'.
std::unordered_map<int, std::string> ReadLabelFile(
    const std::string& file_path);

// Reads file content to the `data` array of given `size`.
void ReadFileToOrDie(const std::string& file_path, char* data, size_t size);

}  // namespace coral
#endif  // LIBCORAL_CORAL_EXAMPLES_FILE_UTILS_H_
