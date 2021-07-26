C++ pipelining APIs
===================

The pipeline API allows you to run inferencing for a segmented model across multiple Edge TPUs.

For more information and a walkthrough of this API, read
`Pipeline a model with multiple Edge TPUs </docs/edgetpu/pipeline/>`_.


`[pipelined_model_runner.h source] <https://github.com/google-coral/libcoral/blob/master/coral/pipeline/pipelined_model_runner.h>`_

.. doxygenclass:: coral::PipelinedModelRunner
   :members:


`[common.h source] <https://github.com/google-coral/libcoral/blob/master/coral/pipeline/common.h>`_

.. doxygenstruct:: coral::PipelineTensor
   :members:

.. doxygenfunction:: coral::FreePipelineTensors

.. doxygenstruct:: coral::SegmentStats
   :members:


`[allocator.h source] <https://github.com/google-coral/libcoral/blob/master/coral/pipeline/allocator.h>`_

.. doxygenclass:: coral::Allocator
   :members:

.. doxygenclass:: coral::Buffer
   :members: