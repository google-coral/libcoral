package(default_visibility = ["//visibility:public"])

cc_library(
    name = "strategy",
    srcs = ["strategy.cc"],
    hdrs = ["strategy.h"],
    deps = [
        ":utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "partition_with_num_ops",
    srcs = ["partition_with_num_ops.cc"],
    deps = [
        ":strategy",
        ":utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
    ],
)

cc_library(
    name = "partitioner",
    srcs = ["partitioner.cc"],
    hdrs = ["partitioner.h"],
    deps = [
        ":strategy",
        ":utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_test(
    name = "partitioner_test",
    srcs = ["partitioner_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":partitioner",
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/schema:schema_utils",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":utils",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "profiling_partition_util",
    testonly = 1,
    srcs = ["profiling_partition_util.cc"],
    hdrs = ["profiling_partition_util.h"],
    deps = [
        ":strategy",
        ":utils",
        "//coral/pipeline:test_utils",
        "//coral/tools:model_pipelining_benchmark_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@glog",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "profiling_partitioner",
    testonly = 1,
    srcs = ["profiling_partitioner.cc"],
    deps = [
        ":profiling_partition_util",
        ":strategy",
        ":utils",
        "//coral/tools:model_pipelining_benchmark_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@glog",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
    ],
)

cc_test(
    name = "profiling_partition_util_test",
    srcs = ["profiling_partition_util_test.cc"],
    data = [
        "@test_data//:models",
    ],
    linkstatic = 1,
    deps = [
        ":profiling_partition_util",
        "//coral:test_main_with_edgetpu",
        "//coral:test_utils",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)
