"""Posenet model benchmark case list."""

POSE_ESTIMATION_MODEL_BENCHMARK_CASES = [
    # PoseNet
    {
        "benchmark_name": "BM_PoseNet_MobileNetV1_075_353_481_WithDecoder",
        "model_path": "posenet/posenet_mobilenet_v1_075_353_481_16_quant_decoder",
    },
    {
        "benchmark_name": "BM_PoseNet_MobileNetV1_075_481_641_WithDecoder",
        "model_path": "posenet/posenet_mobilenet_v1_075_481_641_16_quant_decoder",
    },
    {
        "benchmark_name": "BM_PoseNet_MobileNetV1_075_721_1281_WithDecoder",
        "model_path": "posenet/posenet_mobilenet_v1_075_721_1281_16_quant_decoder",
    },
    # MobileNet BodyPix
    {
        "benchmark_name": "BM_Bodypix_MobileNetV1_075_512_512_WithDecoder",
        "model_path": "posenet/bodypix_mobilenet_v1_075_512_512_16_quant_decoder",
    },
    # MoveNet
    {
        "benchmark_name": "BM_MovenetLightning",
        "model_path": "movenet_single_pose_lightning_ptq",
    },
    {
        "benchmark_name": "BM_MovenetThunder",
        "model_path": "movenet_single_pose_thunder_ptq",
    },
]
