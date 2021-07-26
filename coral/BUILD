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

load("//coral:model_benchmark_cases.bzl", "LSTM_MODEL_BENCHMARK_CASES", "MODEL_BENCHMARK_CASES")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files([
    "classification_model_test_boilerplate.cc",
    "models_benchmark_lib.cc",
])

cc_library(
    name = "error_reporter",
    srcs = [
        "error_reporter.cc",
    ],
    hdrs = [
        "error_reporter.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
    ],
)

cc_test(
    name = "error_reporter_test",
    srcs = [
        "error_reporter_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":error_reporter",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bbox",
    hdrs = [
        "bbox.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@glog",
    ],
)

cc_test(
    name = "bbox_test",
    srcs = [
        "bbox_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":bbox",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "test_utils.h",
    ],
    deps = [
        ":bbox",
        ":tflite_utils",
        "//coral/classification:adapter",
        "//coral/detection:adapter",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "tflite_utils",
    srcs = [
        "tflite_utils.cc",
    ],
    hdrs = [
        "tflite_utils.h",
    ],
    deps = [
        "//coral/pose_estimation:posenet_decoder_op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "test_utils_test",
    srcs = [
        "test_utils_test.cc",
    ],
    data = [
        "@test_data//:images",
    ],
    linkstatic = 1,
    deps = [
        ":test_main_with_edgetpu",
        ":test_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "tflite_utils_test",
    srcs = [
        "tflite_utils_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//:models",
        "@test_data//invalid_models:models",
    ],
    linkstatic = 1,
    deps = [
        ":error_reporter",
        ":test_main_with_edgetpu",
        ":test_utils",
        ":tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "model_benchmark_reporter",
    testonly = 1,
    srcs = [
        "model_benchmark_reporter.cc",
    ],
    hdrs = [
        "model_benchmark_reporter.h",
    ],
    deps = [
        "@com_github_google_benchmark//:benchmark",
    ],
)

cc_library(
    name = "benchmark_main",
    testonly = 1,
    srcs = [
        "benchmark_main.cc",
    ],
    deps = [
        ":model_benchmark_reporter",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "benchmark_main_with_edgetpu",
    testonly = 1,
    deps = [
        ":benchmark_main",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_main_with_edgetpu",
    testonly = 1,
    deps = [
        ":test_main",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)

cc_test(
    name = "segmentation_models_test",
    srcs = [
        "segmentation_models_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_binary(
    name = "cocompiled_models_benchmark",
    testonly = 1,
    srcs = [
        "cocompiled_models_benchmark.cc",
    ],
    data = [
        "@test_data//:models",
        "@test_data//cocompilation:models",
    ],
    deps = [
        "//coral:benchmark_main_with_edgetpu",
        "//coral:test_utils",
        "@com_github_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "inference_repeatability_test",
    timeout = "long",
    srcs = [
        "inference_repeatability_test.cc",
    ],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "model_loading_stress_test",
    timeout = "long",
    srcs = [
        "model_loading_stress_test.cc",
    ],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "inference_stress_test",
    timeout = "long",
    srcs = [
        "inference_stress_test.cc",
    ],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "multiple_tpus_inference_stress_test",
    srcs = [
        "multiple_tpus_inference_stress_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

[cc_library(
    name = "%s_lib" % case.get("benchmark_name").lower(),
    testonly = 1,
    srcs = ["//coral:models_benchmark_lib.cc"],
    local_defines = [
        "ARG_BENCHMARK_NAME=%s" % case.get("benchmark_name"),
        "ARG_TFLITE_CPU_FILEPATH=%s.tflite" % case.get("model_path"),
        "ARG_TFLITE_EDGETPU_FILEPATH=%s_edgetpu.tflite" % case.get("model_path"),
        "ARG_RUN_CPU_MODEL=%s" % int(case.get("run_cpu_model", True)),
        "ARG_RUN_EDGETPU_MODEL=%s" % int(case.get("run_edgetpu_model", True)),
    ],
    deps = [
        "//coral:test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@glog",
    ],
    alwayslink = 1,
) for case in MODEL_BENCHMARK_CASES + LSTM_MODEL_BENCHMARK_CASES]

cc_binary(
    name = "models_benchmark",
    testonly = 1,
    srcs = [],
    data = [
        "@test_data//:models",
    ],
    deps = [
        "//coral:benchmark_main_with_edgetpu",
    ] + [":%s_lib" % case.get("benchmark_name").lower() for case in MODEL_BENCHMARK_CASES],
)

# To run this test with USB connected Edge TPU, set
# USB_ENABLE_BULK_DESCRIPTORS_FROM_DEVICE=1 and reset this variable afterwards.
cc_binary(
    name = "lstm_models_benchmark",
    testonly = 1,
    srcs = [],
    data = [
        "@test_data//:models",
    ],
    deps = [
        "//coral:benchmark_main_with_edgetpu",
    ] + [":%s_lib" % case.get("benchmark_name").lower() for case in LSTM_MODEL_BENCHMARK_CASES],
)

cc_binary(
    name = "single_model_benchmark",
    testonly = 1,
    srcs = ["single_model_benchmark.cc"],
    data = [],
    deps = [
        "//coral:benchmark_main_with_edgetpu",
        "//coral:test_utils",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
