# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
workspace(name = "libcoral")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Configure libedgetpu and downstream libraries (TF and Crosstool).
local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
)

load("@libedgetpu//:workspace.bzl", "libedgetpu_dependencies")
libedgetpu_dependencies()

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

load("@coral_crosstool//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool", cpp_version = "c++17")

# External Dependencies
http_archive(
    name = "com_google_glog",
    sha256 = "6fc352c434018b11ad312cd3b56be3597b4c6b88480f7bd4e18b3a3b2cf961aa",
    strip_prefix = "glog-3ba8976592274bc1f907c402ce22558011d6fc5e",
    urls = [
        "https://github.com/google/glog/archive/3ba8976592274bc1f907c402ce22558011d6fc5e.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])
exports_files(['CMakeLists.txt'])
load(':bazel/glog.bzl', 'glog_library')
glog_library(with_gflags=0)
""",
)

http_archive(
  name = "com_github_google_benchmark",
  sha256 = "6e40ccab16a91a7beff4b5b640b84846867e125ebce6ac0fe3a70c5bae39675f",
  strip_prefix = "benchmark-16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
  urls = [
    "https://github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.tar.gz"
  ],
)

# Local repos
new_local_repository(
    name = "test_data",
    path = "test_data",
    build_file = "test_data/BUILD"
)

new_local_repository(
    name = "glog",
    path = "third_party/glog",
    build_file = "third_party/glog/BUILD",
)

new_local_repository(
    name = "system_linux",
    build_file = "third_party/system_linux/BUILD",
    path = "/usr/system_libs",
)

