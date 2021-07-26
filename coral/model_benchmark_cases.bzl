"""Model benchmark case list."""

MODEL_BENCHMARK_CASES = [
    {
        "benchmark_name": "BM_MobileNetV1",
        "model_path": "mobilenet_v1_1.0_224_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV1_25",
        "model_path": "mobilenet_v1_0.25_128_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV1_50",
        "model_path": "mobilenet_v1_0.5_160_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV1_75",
        "model_path": "mobilenet_v1_0.75_192_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV1_L2Norm",
        "model_path": "mobilenet_v1_1.0_224_l2norm_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV2",
        "model_path": "mobilenet_v2_1.0_224_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatPlant",
        "model_path": "mobilenet_v2_1.0_224_inat_plant_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatInsect",
        "model_path": "mobilenet_v2_1.0_224_inat_insect_quant",
    },
    {
        "benchmark_name": "BM_MobileNetV2INatBird",
        "model_path": "mobilenet_v2_1.0_224_inat_bird_quant",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV1",
        "model_path": "ssd_mobilenet_v1_coco_quant_postprocess",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV2",
        "model_path": "ssd_mobilenet_v2_coco_quant_postprocess",
    },
    {
        "benchmark_name": "BM_FaceSsd",
        "model_path": "ssd_mobilenet_v2_face_quant_postprocess",
    },
    {
        "benchmark_name": "BM_InceptionV1",
        "model_path": "inception_v1_224_quant",
    },
    {
        "benchmark_name": "BM_InceptionV2",
        "model_path": "inception_v2_224_quant",
    },
    {
        "benchmark_name": "BM_InceptionV3",
        "model_path": "inception_v3_299_quant",
    },
    {
        "benchmark_name": "BM_InceptionV4",
        "model_path": "inception_v4_299_quant",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuSmall",
        "model_path": "efficientnet-edgetpu-S_quant",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuMedium",
        "model_path": "efficientnet-edgetpu-M_quant",
    },
    {
        "benchmark_name": "BM_EfficientNetEdgeTpuLarge",
        "model_path": "efficientnet-edgetpu-L_quant",
    },
    {
        "benchmark_name": "BM_Deeplab513Mv2Dm1_WithArgMax",
        "model_path": "deeplabv3_mnv2_pascal_quant",
    },
    {
        "benchmark_name": "BM_Deeplab513Mv2Dm05_WithArgMax",
        "model_path": "deeplabv3_mnv2_dm05_pascal_quant",
    },
    {
        "benchmark_name": "BM_KerasPostTrainingQuantizedUnetMv2128",
        "model_path": "keras_post_training_unet_mv2_128_quant",
    },
    {
        "benchmark_name": "BM_KerasPostTrainingQuantizedUnetMv2256",
        "model_path": "keras_post_training_unet_mv2_256_quant",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV1FineTunedPet",
        "model_path": "ssd_mobilenet_v1_fine_tuned_pet",
    },
    {
        "benchmark_name": "BM_PostTrainingQuantizedTf2KerasMobileNetV1",
        "model_path": "tf2_mobilenet_v1_1.0_224_ptq",
    },
    {
        "benchmark_name": "BM_PostTrainingQuantizedTf2KerasMobileNetV2",
        "model_path": "tf2_mobilenet_v2_1.0_224_ptq",
    },
    {
        "benchmark_name": "BM_PostTrainingQuantizedTf2KerasMobileNetV3EdgeTpu",
        "model_path": "tf2_mobilenet_v3_edgetpu_1.0_224_ptq",
    },
    {
        "benchmark_name": "BM_SsdLiteMobileDet",
        "model_path": "ssdlite_mobiledet_coco_qat_postprocess",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV1_NoNms",
        "model_path": "ssd_mobilenet_v1_coco_quant_no_nms",
    },
    {
        "benchmark_name": "BM_SsdMobileNetV2_NoNms",
        "model_path": "ssd_mobilenet_v2_coco_quant_no_nms",
    },
    {
        "benchmark_name": "BM_Tf2Resnet50ImagenetPtq",
        "model_path": "tfhub_tf2_resnet_50_imagenet_ptq",
    },
    {
        "benchmark_name": "BM_Tf2SsdMobilenetV2Ptq",
        "model_path": "tf2_ssd_mobilenet_v2_coco17_ptq",
    },
    {
        "benchmark_name": "BM_Tf2SsdMobilenetV1Fpn640x640Ptq",
        "model_path": "tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq",
    },
    {
        "benchmark_name": "BM_Deeplab513MvEdgetpuDm075_WithArgMax",
        "model_path": "deeplab_mobilenet_edgetpu_slim_cityscapes_quant",
    },
    {
        "benchmark_name": "BM_TfhubPopularUsProductsPtq",
        "model_path": "tfhub_tf1_popular_us_products_ptq",
    },
    {
        "benchmark_name": "BM_TfhubPopularUsProductsPtqFcSplit",
        "model_path": "tfhub_tf1_popular_us_products_ptq_fc_split",
    },
    {
        "benchmark_name": "BM_EfficientDetLite0_320",
        "model_path": "efficientdet_lite0_320_ptq",
    },
    {
        "benchmark_name": "BM_EfficientDetLite1_384",
        "model_path": "efficientdet_lite1_384_ptq",
    },
    {
        "benchmark_name": "BM_EfficientDetLite2_448",
        "model_path": "efficientdet_lite2_448_ptq",
    },
    {
        "benchmark_name": "BM_EfficientDetLite3_512",
        "model_path": "efficientdet_lite3_512_ptq",
    },
    {
        "benchmark_name": "BM_EfficientDetLite3x_640",
        "model_path": "efficientdet_lite3x_640_ptq",
        "target_device": "{\\\"form\\\": \\\"mpcie\\\"}",
    },
]

LSTM_MODEL_BENCHMARK_CASES = [
    {
        "benchmark_name": "BM_KerasLstmMnist",
        "model_path": "keras_lstm_mnist_ptq",
    },
]
