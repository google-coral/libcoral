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

load("//coral/pose_estimation:model_benchmark_cases.bzl", "POSE_ESTIMATION_MODEL_BENCHMARK_CASES")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_test(
    name = "movenet_test",
    srcs = [
        "movenet_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//:models",
        "@test_data//:reference_results",
    ],
    linkstatic = 1,
    deps = [
        ":test_utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "bodypix_test",
    srcs = [
        "bodypix_test.cc",
    ],
    data = [
        "@test_data//posenet:images",
        "@test_data//posenet:models",
        "@test_data//posenet:reference_results",
    ],
    linkstatic = 1,
    deps = [
        ":posenet_decoder_op",
        ":test_utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "posenet_test",
    srcs = [
        "posenet_test.cc",
    ],
    data = [
        "@test_data//posenet:images",
        "@test_data//posenet:models",
        "@test_data//posenet:reference_results",
    ],
    linkstatic = 1,
    deps = [
        ":posenet_decoder_op",
        ":test_utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
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
) for case in POSE_ESTIMATION_MODEL_BENCHMARK_CASES]

cc_binary(
    name = "models_benchmark",
    testonly = 1,
    data = [
        "@test_data//:models",
        "@test_data//posenet:models",
    ],
    deps = [
        "//coral:benchmark_main_with_edgetpu",
    ] + [":%s_lib" % case.get("benchmark_name").lower() for case in POSE_ESTIMATION_MODEL_BENCHMARK_CASES],
)

cc_test(
    name = "posenet_decoder_test",
    timeout = "long",
    srcs = [
        "posenet_decoder_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":posenet_decoder",
        "//coral:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
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
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@libedgetpu//tflite/public:edgetpu",
    ],
)

cc_library(
    name = "posenet_decoder_op",
    srcs = [
        "posenet_decoder_op.cc",
    ],
    hdrs = [
        "posenet_decoder_op.h",
    ],
    deps = [
        ":posenet_decoder",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "posenet_decoder",
    srcs = [
        "posenet_decoder.cc",
    ],
    hdrs = [
        "posenet_decoder.h",
    ],
)

cc_binary(
    name = "posenet_decoder.so",
    srcs = ["posenet_decoder_tflite_plugin.cc"],
    linkopts = [
        "-Wl,-soname,posenet_decoder.so.1",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":posenet_decoder_op",
        "@org_tensorflow//tensorflow/lite:kernel_api",
        "@org_tensorflow//tensorflow/lite:util",
    ],
)
