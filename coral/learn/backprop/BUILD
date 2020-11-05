package(default_visibility = ["//visibility:public"])

cc_library(
    name = "layers",
    srcs = ["layers.cc"],
    hdrs = ["layers.h"],
    deps = [
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "layers_test",
    srcs = ["layers_test.cc"],
    linkstatic = 1,
    deps = [
        ":layers",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":multi_variate_normal_distribution",
        ":softmax_regression_model",
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.cc"],
    linkstatic = 1,
    deps = [
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "softmax_regression_model",
    srcs = ["softmax_regression_model.cc"],
    hdrs = ["softmax_regression_model.h"],
    deps = [
        ":layers",
        ":multi_variate_normal_distribution",
        "//coral:error_reporter",
        "//coral/learn:utils",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "softmax_regression_model_test",
    srcs = ["softmax_regression_model_test.cc"],
    linkstatic = 1,
    deps = [
        ":layers",
        ":softmax_regression_model",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
        "@glog",
    ],
)

cc_library(
    name = "multi_variate_normal_distribution",
    srcs = ["multi_variate_normal_distribution.cc"],
    hdrs = ["multi_variate_normal_distribution.h"],
    deps = [
        "@eigen_archive//:eigen",
        "@glog",
    ],
)

cc_test(
    name = "multi_variate_normal_distribution_test",
    srcs = ["multi_variate_normal_distribution_test.cc"],
    linkstatic = 1,
    deps = [
        ":multi_variate_normal_distribution",
        "@com_google_googletest//:gtest_main",
        "@glog",
    ],
)

cc_binary(
    name = "softmax_regression_model_benchmark",
    testonly = 1,
    srcs = [
        "softmax_regression_model_benchmark.cc",
    ],
    linkstatic = 1,
    deps = [
        ":layers",
        ":softmax_regression_model",
        ":test_utils",
        "//coral:benchmark_main",
        "@com_github_google_benchmark//:benchmark",
        "@glog",
    ],
)
