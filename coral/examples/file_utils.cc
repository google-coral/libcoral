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
  std::unordered_map<int, std::string> labels;
  std::ifstream file(file_path.c_str());
  CHECK(file) << "Cannot open " << file_path;

  std::string line;
  while (std::getline(file, line)) {
    absl::RemoveExtraAsciiWhitespace(&line);
    std::vector<std::string> fields =
        absl::StrSplit(line, absl::MaxSplits(' ', 1));
    if (fields.size() == 2) {
      int label_id;
      CHECK(absl::SimpleAtoi(fields[0], &label_id));
      const std::string& label_name = fields[1];
      labels[label_id] = label_name;
    }
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
