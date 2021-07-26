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
    name = "common",
    hdrs = ["common.h"],
    deps = [
        ":allocator",
        "@glog",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    deps = [],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":allocator",
        ":common",
        ":pipelined_model_runner",
        "//coral:error_reporter",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
    ],
)

cc_library(
    name = "pipelined_model_runner",
    srcs = ["pipelined_model_runner.cc"],
    hdrs = ["pipelined_model_runner.h"],
    deps = [
        ":allocator",
        ":common",
        "//coral:tflite_utils",
        "//coral/pipeline/internal:default_allocator",
        "//coral/pipeline/internal:memory_pool_allocator",
        "//coral/pipeline/internal:segment_runner",
        "//coral/pipeline/internal:thread_safe_queue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "pipelined_model_runner_test",
    srcs = ["pipelined_model_runner_test.cc"],
    data = [
        "@test_data//:models",
        "@test_data//pipeline:models",
    ],
    linkstatic = 1,
    deps = [
        ":common",
        ":pipelined_model_runner",
        ":test_utils",
        "//coral:error_reporter",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "models_test_main",
    testonly = 1,
    srcs = ["models_test_main.cc"],
    deps = [
        ":common",
        ":pipelined_model_runner",
        ":test_utils",
        "//coral:error_reporter",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "//coral/pipeline/internal:default_allocator",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "models_test",
    args = [
        "--model_names=inception_v3_299_quant,inception_v4_299_quant",
    ],
    data = [
        "@test_data//:models",
        "@test_data//pipeline:models",
    ],
    linkstatic = 1,
    deps = [
        ":models_test_main",
    ],
)

cc_library(
    name = "detection_models_test_lib",
    testonly = 1,
    srcs = [
        "detection_models_test_lib.cc",
    ],
    hdrs = [
        "detection_models_test_lib.h",
    ],
    deps = [
        ":common",
        ":pipelined_model_runner",
        ":test_utils",
        "//coral:error_reporter",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "//coral/detection:adapter",
        "//coral/pipeline/internal:default_allocator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "detection_models_test",
    srcs = [
        "detection_models_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//pipeline:models",
    ],
    linkstatic = 1,
    deps = [
        ":detection_models_test_lib",
        ":test_utils",
        "//coral:test_main_with_edgetpu",
    ],
)
