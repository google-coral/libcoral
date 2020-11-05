package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "tflite_graph_util",
    srcs = ["tflite_graph_util.cc"],
    hdrs = ["tflite_graph_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_conversion_utils",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)

cc_binary(
    name = "join_tflite_models",
    srcs = ["join_tflite_models.cc"],
    deps = [
        ":tflite_graph_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@glog",
    ],
)

cc_test(
    name = "tflite_graph_util_test",
    srcs = ["tflite_graph_util_test.cc"],
    data = [
        "@test_data//tools:models",
    ],
    linkstatic = 1,
    deps = [
        ":tflite_graph_util",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@glog",
    ],
)

cc_binary(
    name = "multiple_tpus_performance_analysis",
    testonly = 1,
    srcs = ["multiple_tpus_performance_analysis.cc"],
    data = [
        "@test_data//:images",
        "@test_data//:models",
    ],
    deps = [
        "//coral:test_utils",
        "//coral:tflite_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)

cc_library(
    name = "model_pipelining_benchmark_util",
    testonly = 1,
    srcs = ["model_pipelining_benchmark_util.cc"],
    hdrs = ["model_pipelining_benchmark_util.h"],
    deps = [
        "//coral:test_utils",
        "//coral:tflite_utils",
        "//coral/pipeline:common",
        "//coral/pipeline:pipelined_model_runner",
        "//coral/pipeline:test_utils",
        "//coral/pipeline:utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "model_pipelining_performance_analysis",
    testonly = 1,
    srcs = ["model_pipelining_performance_analysis.cc"],
    deps = [
        ":model_pipelining_benchmark_util",
        "//coral/pipeline:test_utils",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)
