"""Test case lists for classification model tests."""

COCOMPILED_CLASSIFICATION_MODEL_TEST_CASES = [
    {
        "model_path": "cocompilation/inception_v1_224_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v1_224_quant_cocompiled_with_inception_v4_299_quant_edgetpu.tflite",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v2_224_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.51",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v3_299_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.5",
        "expected_topk_label": "282",
    },
    {
        "model_path": "cocompilation/inception_v3_299_quant_cocompiled_with_inception_v4_299_quant_edgetpu.tflite",
        "score_threshold": "0.5",
        "expected_topk_label": "282",
    },
    {
        "model_path": "cocompilation/inception_v4_299_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v4_299_quant_cocompiled_with_inception_v1_224_quant_edgetpu.tflite",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v4_299_quant_cocompiled_with_inception_v3_299_quant_edgetpu.tflite",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/inception_v4_299_quant_cocompiled_with_mobilenet_v1_0.25_128_quant_edgetpu.tflite",
        "score_threshold": "0.25",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.25",
        "expected_topk_label": "283",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_inception_v4_299_quant_edgetpu.tflite",
        "score_threshold": "0.25",
        "expected_topk_label": "283",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.25_128_quant_cocompiled_with_mobilenet_v1_0.5_160_quant_edgetpu.tflite",
        "score_threshold": "0.25",
        "expected_topk_label": "283",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.5_160_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.51",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.5_160_quant_cocompiled_with_mobilenet_v1_0.25_128_quant_edgetpu.tflite",
        "score_threshold": "0.51",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_0.75_192_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.35",
        "expected_topk_label": "283",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_1.0_224_quant_cocompiled_with_3quant_edgetpu.tflite",
        "score_threshold": "0.78",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/mobilenet_v1_1.0_224_quant_cocompiled_with_mobilenet_v2_1.0_224_quant_edgetpu.tflite",
        "score_threshold": "0.78",
        "expected_topk_label": "286",
    },
    {
        "model_path": "cocompilation/mobilenet_v2_1.0_224_quant_cocompiled_with_mobilenet_v1_1.0_224_quant_edgetpu.tflite",
        "score_threshold": "0.78",
        "expected_topk_label": "286",
    },
]

# List of pairs of CPU and Edge TPU models.
# It's expected that one pair of CPU and Edge TPU models can share the
# same set of test parameters.
_CPU_EDGETPU_MODEL_PAIR_CASES = [
    {
        "model_path": "mobilenet_v1_1.0_224_quant",
        "score_threshold": "0.78",
        "expected_topk_label": "286",
    },
    {
        "model_path": "mobilenet_v1_0.25_128_quant",
        "score_threshold": "0.25",
        "expected_topk_label": "283",
    },
    {
        "model_path": "mobilenet_v1_0.5_160_quant",
        "score_threshold": "0.51",
        "expected_topk_label": "286",
    },
    {
        "model_path": "mobilenet_v1_0.75_192_quant",
        "score_threshold": "0.35",
        "expected_topk_label": "283",
    },
    {
        "model_path": "mobilenet_v2_1.0_224_quant",
        "score_threshold": "0.7",
        "expected_topk_label": "286",
    },
    {
        "model_path": "inception_v1_224_quant",
        "score_threshold": "0.37",
        "expected_topk_label": "282",
    },
    {
        "model_path": "inception_v2_224_quant",
        "score_threshold": "0.51",
        "expected_topk_label": "286",
    },
    {
        "model_path": "inception_v3_299_quant",
        "score_threshold": "0.5",
        "expected_topk_label": "282",
    },
    {
        "model_path": "inception_v4_299_quant",
        "score_threshold": "0.3",
        "expected_topk_label": "286",
    },
    {
        "model_path": "mobilenet_v2_1.0_224_inat_plant_quant",
        "image_path": "sunflower.bmp",
        "score_threshold": "0.8",
        "k": "1",
        "expected_topk_label": "1680",
    },
    {
        "model_path": "mobilenet_v2_1.0_224_inat_insect_quant",
        "image_path": "dragonfly.bmp",
        "score_threshold": "0.2",
        "k": "1",
        "expected_topk_label": "912",
    },
    {
        "model_path": "mobilenet_v2_1.0_224_inat_bird_quant",
        "image_path": "bird.bmp",
        "score_threshold": "0.5",
        "k": "1",
        "expected_topk_label": "659",
    },
    {
        "model_path": "efficientnet-edgetpu-S_quant",
        "effective_scale": "1.608448",
        "effective_means": "-83.70668800000001,-83.70668800000001,-83.70668800000001",
        "score_threshold": "0.35",
        "expected_topk_label": "286",
    },
    {
        "model_path": "efficientnet-edgetpu-M_quant",
        "effective_scale": "1.547392",
        "effective_means": "-72.61356800000001,-72.61356800000001,-72.61356800000001",
        "score_threshold": "0.5",
        "expected_topk_label": "286",
    },
    {
        "model_path": "efficientnet-edgetpu-L_quant",
        "effective_scale": "1.59488",
        "effective_means": "-78.73952,-78.73952,-78.73952",
        "score_threshold": "0.45",
        "expected_topk_label": "286",
    },
    {
        "model_path": "tf2_mobilenet_v1_1.0_224_ptq",
        "score_threshold": "0.8",
        "expected_topk_label": "286",
    },
    {
        "model_path": "tf2_mobilenet_v2_1.0_224_ptq",
        "score_threshold": "0.6",
        "expected_topk_label": "286",
    },
    {
        "model_path": "tf2_mobilenet_v3_edgetpu_1.0_224_ptq",
        "effective_scale": "1.4810551404953003",
        "effective_means": "-55.169782280921936,-55.169782280921936,-55.169782280921936",
        "score_threshold": "0.35",
        "expected_topk_label": "286",
    },
    {
        "model_path": "tfhub_tf2_resnet_50_imagenet_ptq",
        "score_threshold": "0.5",
        "expected_topk_label": "285",  # this model's object label starts from 0
    },
    {
        "model_path": "tfhub_tf1_popular_us_products_ptq",
        "image_path": "missvickie_potato_chips.bmp",
        "score_threshold": "0.7",
        "expected_topk_label": "77109",  # See https://www.gstatic.com/aihub/tfhub/labelmaps/popular_us_products_V1_labelmap.csv
    },
    {
        "model_path": "tfhub_tf1_popular_us_products_ptq_fc_split",
        "image_path": "missvickie_potato_chips.bmp",
        "score_threshold": "0.7",
        "expected_topk_label": "77109",  # See https://www.gstatic.com/aihub/tfhub/labelmaps/popular_us_products_V1_labelmap.csv
    },
]

LSTM_MNIST_MODEL_TEST_CASES = [
    {
        "model_path": case.get("model_path") + model_suffix,
        "image_path": case.get("image_path", "cat.bmp"),
        "score_threshold": case.get("score_threshold"),
        "expected_topk_label": case.get("expected_topk_label"),
        "effective_scale": "1",
        "effective_means": "0",
    }
    for model_suffix in [".tflite", "_edgetpu.tflite"]
    for case in [
        {
            "model_path": "keras_lstm_mnist_ptq",
            "image_path": "mnist_nine.bmp",
            "score_threshold": "0.7",
            "expected_topk_label": "9",
        },
    ]
]

# Expand CPU / Edge TPU model pairs to a full list of individually compiled models.
INDIVIDUALLY_COMPILED_CLASSIFICATION_MODEL_TEST_CASES = [
    {
        "model_path": case.get("model_path") + model_suffix,
        "image_path": case.get("image_path", "cat.bmp"),
        "effective_scale": case.get("effective_scale", "1"),
        "effective_means": case.get("effective_means", "0,0,0"),
        "rgb2bgr": case.get("rgb2bgr", "false"),
        "score_threshold": case.get("score_threshold"),
        "k": case.get("k", "3"),
        "expected_topk_label": case.get("expected_topk_label"),
    }
    for model_suffix in [".tflite", "_edgetpu.tflite"]
    for case in _CPU_EDGETPU_MODEL_PAIR_CASES
]

CLASSIFICATION_MODEL_TEST_CASES = COCOMPILED_CLASSIFICATION_MODEL_TEST_CASES + INDIVIDUALLY_COMPILED_CLASSIFICATION_MODEL_TEST_CASES
