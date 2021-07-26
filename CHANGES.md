# libcoral changes


## Grouper release

*  API changes for `PipelinedModelRunner`:
  *  `Push()` and `Pop()` now return either `absl::OkStatus` or
     `absl::InternalError`, instead of True/False.
  *  Added tensor `name` to `PipelineTensor`.
  *  `GetInputTensorNames()` and `GetInputTensor()` moved to `tflite_utils.h`.
  * `GetDetectionResults()` now supports SSD models with different orders in
     the output tensor.

*  Updated profiling-based partitioner:
  *  Renamed executable to `partition_with_profiling`.
  *  Added support for SSD models, and other models with large CPU segments
     and/or graph branches. (This also requires v16 of the Edge TPU Compiler.)
  *  Automatically enables the [`search_delegate`
  option](https://coral.ai/docs/edgetpu/compiler#usage) (add in v16 of the Edge
  TPU Compiler), so that if the compiler first fails to compile the model, it
  continues to search for a portion of the graph that can compile.
  *  Added flags:
      *  `delegate_search_step`: Same as the `delegate_search_step` option
      added in v16 of the Edge TPU Compiler.
      *  `partition_search_step`: Similar to the `delegate_search_step`
      option, but applied to the delegate search for each segment (rather than
      the entire pipelined graph).
      *  `initial_lower_bound_ns` and `initial_upper_bound_ns`: The known
      smallest/largest latency among your model's segments. These are otherwise
      calculated in the tool by benchmarking the heuristic-based segments from
      the Edge TPU Compiler.

*  Added `split_fc` tool to pre-process tflite models with fully-connected
layers that are too big for the Edge TPU Compiler. For example, the compiler is
unable to compile a layer that has 100,000 output classes due to the size of the
weights matrix applied to the fully-connected layer (this operation would be cut
from the Edge TPU delegate and instead execute on the CPU). So the `split_fc`
tool divides the weights into smaller blocks using block-wise matrix
multiplication (you can control the ratio of the split operation). The revised
`.tflite` file output by `split_fc` can then be passed to the Edge TPU Compiler.

*  Added `append_recurrent_links` tool, which helps you create recurrent
networks for the Edge TPU with a hidden saved state. *Without this tool*,
creating a recurrent network that can compile for the Edge TPU typically
requires that your model output the saved state and then you must pass back
the saved state into your model manually. By instead passing such a model to the
`append_recurrent_links` tool, you can make that saved state hidden again so
your application code can focus on the final output.


## Frogfish release

*   Initial libcoral release
