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

cc_test(
    name = "dmabuf_devboard_test",
    srcs = [
        "dmabuf_devboard_test.cc",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@system_linux//:gstallocators",
        "@system_linux//:gstreamer",
    ],
)

cc_test(
    name = "model_pipelining_dmabuf_devboard_test",
    srcs = [
        "model_pipelining_dmabuf_devboard_test.cc",
    ],
    linkstatic = 1,
    deps = [
        "//coral:error_reporter",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral/pipeline:pipelined_model_runner",
        "//coral/pipeline:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@system_linux//:gstallocators",
        "@system_linux//:gstreamer",
    ],
)
