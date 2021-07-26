# Copyright 2019-2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "aligned_alloc",
    hdrs = [
        "aligned_alloc.h",
    ],
)

cc_library(
    name = "thread_safe_queue",
    hdrs = ["thread_safe_queue.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "default_allocator",
    hdrs = ["default_allocator.h"],
    deps = ["//coral/pipeline:allocator"],
)

cc_library(
    name = "memory_pool_allocator",
    srcs = ["memory_pool_allocator.cc"],
    hdrs = ["memory_pool_allocator.h"],
    deps = [
        ":aligned_alloc",
        "//coral/pipeline:allocator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
        "@glog",
    ],
)

cc_test(
    name = "memory_pool_allocator_test",
    srcs = ["memory_pool_allocator_test.cc"],
    linkstatic = 1,
    deps = [
        ":memory_pool_allocator",
        "//coral:test_main",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "segment_runner",
    srcs = ["segment_runner.cc"],
    hdrs = ["segment_runner.h"],
    deps = [
        ":thread_safe_queue",
        "//coral/pipeline:allocator",
        "//coral/pipeline:common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
    ],
)

cc_test(
    name = "segment_runner_test",
    srcs = ["segment_runner_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":default_allocator",
        ":segment_runner",
        "//coral:error_reporter",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral/pipeline:allocator",
        "//coral/pipeline:test_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
