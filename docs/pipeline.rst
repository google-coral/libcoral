C++ pipeline API
================

The pipeline API allows you to run inferencing for a segmented model across multiple Edge TPUs.

For more information and a walkthrough of this API, read
`Pipeline a model with multiple Edge TPUs </docs/edgetpu/pipeline/>`_.

Source code and header files are at
https://github.com/google-coral/edgetpu/tree/master/src/cpp/pipeline/.

``coral/pipeline/utils.h``

.. doxygenfile:: pipeline/utils.h

.. doxygenclass:: coral::Allocator
   :members:


.. doxygenclass:: coral::PipelinedModelRunner
   :members:


.. doxygenstruct:: coral::PipelineTensor
   :members:


.. doxygenstruct:: coral::SegmentStats
   :members: