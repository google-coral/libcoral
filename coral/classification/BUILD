load(
    "//coral/classification:model_test_cases.bzl",
    "CLASSIFICATION_MODEL_TEST_CASES",
)
load(
    "//coral:test_utils.bzl",
    "model_path_to_test_name",
)

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
    linkstatic = 1,
    deps = [
        ":adapter",
        "//coral:test_main",
        "@com_google_googletest//:gtest",
        "@glog",
    ],
)

[cc_library(
    name = "%s_lib" % model_path_to_test_name(case.get("model_path")),
    testonly = 1,
    srcs = ["//coral:classification_model_test_lib.cc"],
    local_defines = [
        "ARG_EFFECTIVE_MEANS=%s" % case.get("effective_means", "0,0,0"),
        "ARG_EFFECTIVE_SCALE=%s" % case.get("effective_scale", "1"),
        "ARG_EXPECTED_TOPK_LABEL=%s" % case.get("expected_topk_label"),
        "ARG_IMAGE_PATH=%s" % case.get("image_path", "cat.bmp"),
        "ARG_K=%s" % case.get("k", "3"),
        "ARG_MODEL_PATH=%s" % case.get("model_path"),
        "ARG_RGB2BGR=%s" % case.get("rgb2bgr", "false"),
        "ARG_SCORE_THRESHOLD=%s" % case.get("score_threshold", "0"),
        "ARG_TEST_NAME=%s" % model_path_to_test_name(case.get("model_path")),
        "ARG_TEST_SUITE_NAME=%s" % ("ClassificationModelEdgeTpuTest" if case.get("model_path").endswith("_edgetpu.tflite") else "ClassificationModelCpuTest",),
    ],
    deps = [
        "//coral:test_utils",
        "@com_google_googletest//:gtest",
    ],
    alwayslink = 1,
) for case in CLASSIFICATION_MODEL_TEST_CASES]

cc_test(
    name = "classification_models_test",
    srcs = [],
    data = [
        "@test_data//:images",
        "@test_data//:models",
        "@test_data//cocompilation:models",
    ],
    linkstatic = 1,
    deps = [
        "//coral:test_main_with_edgetpu",
    ] + [":%s_lib" % model_path_to_test_name(case.get("model_path")) for case in CLASSIFICATION_MODEL_TEST_CASES],
)
