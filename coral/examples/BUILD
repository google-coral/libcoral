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
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@glog",
    ],
)

cc_binary(
    name = "two_models_one_tpu",
    srcs = ["two_models_one_tpu.cc"],
    deps = [
        ":file_utils",
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "two_models_two_tpus_threaded",
    srcs = ["two_models_two_tpus_threaded.cc"],
    deps = [
        ":file_utils",
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "classify_image",
    srcs = ["classify_image.cc"],
    deps = [
        ":file_utils",
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

cc_binary(
    name = "backprop_last_layer",
    srcs = ["backprop_last_layer.cc"],
    deps = [
        ":file_utils",
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "//coral/learn/backprop:softmax_regression_model",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@eigen_archive//:eigen3",
        "@flatbuffers",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "model_pipelining",
    srcs = ["model_pipelining.cc"],
    deps = [
        "//coral/pipeline:pipelined_model_runner",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "lstpu",
    srcs = ["lstpu.cc"],
    deps = [
        "@libedgetpu//tflite/public:edgetpu_c",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)
