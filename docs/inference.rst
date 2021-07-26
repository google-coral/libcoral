C++ inferencing APIs
====================

TensorFlow Lite utilities
-------------------------

The following APIs simplify your code when working with a `tflite::Interpreter
<https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter>`_.

`[tflite_utils.h source] <https://github.com/google-coral/libcoral/blob/master/coral/tflite_utils.h>`_

.. doxygenfile:: tflite_utils.h


Image classification
--------------------

Use the following APIs with image classification models.

`[adapter.h source] <https://github.com/google-coral/libcoral/blob/master/coral/classification/adapter.h>`_

.. doxygenfile:: classification/adapter.h


Object detection
----------------

Use the following APIs with object detection models.

`[adapter.h source] <https://github.com/google-coral/libcoral/blob/master/coral/detection/adapter.h>`_

.. doxygenfile:: detection/adapter.h


`[bbox.h source] <https://github.com/google-coral/libcoral/blob/master/coral/bbox.h>`_

.. doxygenfile:: bbox.h