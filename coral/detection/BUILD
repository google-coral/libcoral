package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

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
    ],
)
