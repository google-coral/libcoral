config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    }
)

cc_library(
    name = "glog",
    defines = select({
        ":windows": ["GLOG_NO_ABBREVIATED_SEVERITIES"],
        "//conditions:default": [],
    }),
    deps = ["@com_google_glog//:glog"],
    visibility = ["//visibility:public"],
)
