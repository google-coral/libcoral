package(default_visibility = ["//visibility:public"])

cc_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        ":allocator",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    deps = [],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":allocator",
        ":common",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        ":allocator",
        ":common",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "pipelined_model_runner",
    srcs = ["pipelined_model_runner.cc"],
    hdrs = ["pipelined_model_runner.h"],
    deps = [
        ":allocator",
        ":common",
        ":utils",
        "//coral/pipeline/internal:default_allocator",
        "//coral/pipeline/internal:memory_pool_allocator",
        "//coral/pipeline/internal:segment_runner",
        "//coral/pipeline/internal:thread_safe_queue",
        "@com_google_absl//absl/memory",
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
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
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
        ":utils",
        "//coral:test_utils",
        "//coral:tflite_utils",
        "//coral/pipeline/internal:default_allocator",
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
