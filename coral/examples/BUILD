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
        "@eigen_archive//:eigen",
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
        "//coral/pipeline:utils",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
