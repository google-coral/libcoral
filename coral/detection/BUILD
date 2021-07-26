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
    name = "adapter",
    srcs = [
        "adapter.cc",
    ],
    hdrs = [
        "adapter.h",
    ],
    deps = [
        "//coral:bbox",
        "//coral:tflite_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "adapter_test",
    srcs = [
        "adapter_test.cc",
    ],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":adapter",
        "//coral:bbox",
        "//coral:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

cc_test(
    name = "models_test",
    srcs = [
        "models_test.cc",
    ],
    data = [
        "@test_data//:images",
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@libedgetpu//tflite/public:edgetpu",
    ],
)
