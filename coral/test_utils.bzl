"""Test utility functions."""

def model_path_to_test_name(model_path):
    """Generates string for test name from model path.

    Args:
      model_path: model path.
    """

    # Remove .tflite extension.
    tmp = model_path.split(".tflite")[0]
    return tmp.replace(".", "_").replace("/", "_").replace("-", "_")
