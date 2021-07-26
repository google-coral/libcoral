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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "strategy",
    srcs = ["strategy.cc"],
    hdrs = ["strategy.h"],
    deps = [
        ":utils",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "partition_with_num_ops",
    srcs = ["partition_with_num_ops.cc"],
    deps = [
        ":strategy",
        ":utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
    ],
)

cc_library(
    name = "parameter_count_based_partitioner",
    srcs = ["parameter_count_based_partitioner.cc"],
    hdrs = ["parameter_count_based_partitioner.h"],
    deps = [
        ":strategy",
        ":utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "parameter_count_based_partitioner_test",
    srcs = ["parameter_count_based_partitioner_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":parameter_count_based_partitioner",
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "profiling_based_partitioner",
    testonly = 1,
    srcs = ["profiling_based_partitioner.cc"],
    hdrs = ["profiling_based_partitioner.h"],
    deps = [
        ":strategy",
        ":utils",
        "//coral/pipeline:test_utils",
        "//coral/tools:model_pipelining_benchmark_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "partition_with_profiling",
    testonly = 1,
    srcs = ["partition_with_profiling.cc"],
    deps = [
        ":profiling_based_partitioner",
        ":strategy",
        ":utils",
        "//coral/tools:model_pipelining_benchmark_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)

cc_test(
    name = "profiling_based_partitioner_test",
    srcs = ["profiling_based_partitioner_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":profiling_based_partitioner",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_library(
    name = "profiling_based_partitioner_ondevice_lib",
    testonly = 1,
    srcs = ["profiling_based_partitioner_ondevice_lib.cc"],
    hdrs = ["profiling_based_partitioner_ondevice_lib.h"],
    deps = [
        ":profiling_based_partitioner",
        "//coral:test_utils",
        "//coral/pipeline:test_utils",
        "//coral/tools:model_pipelining_benchmark_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "profiling_based_partitioner_ondevice_test",
    size = "large",
    srcs = ["profiling_based_partitioner_ondevice_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":profiling_based_partitioner_ondevice_lib",
        "//coral:test_main_with_edgetpu",
        "@com_google_googletest//:gtest",
    ],
)
