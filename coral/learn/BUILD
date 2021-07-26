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
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//coral:tflite_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_conversion_utils",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)

cc_test(
    name = "utils_test",
    srcs = [
        "utils_test.cc",
    ],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "imprinting_engine",
    srcs = [
        "imprinting_engine.cc",
    ],
    hdrs = [
        "imprinting_engine.h",
    ],
    deps = [
        ":utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)

cc_test(
    name = "imprinting_engine_test",
    srcs = [
        "imprinting_engine_test.cc",
    ],
    data = [
        "@test_data//:models",
        "@test_data//imprinting:images",
    ],
    linkstatic = 1,
    deps = [
        ":imprinting_engine",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)
