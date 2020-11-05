package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

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
