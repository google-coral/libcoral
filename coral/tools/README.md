# Coral developer tools

This directory holds a variety of tools to help you develop and evaluate
models for the Coral Edge TPU.


## Build the tools

To compile these tools, you first need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
(optionally, but we recommend) [Docker](https://docs.docker.com/install/).

Then clone this repo with submodules and build the tools:

```
git clone --recurse-submodules https://github.com/google-coral/libcoral.git

cd libcoral

make DOCKER_IMAGE="ubuntu:18.04" DOCKER_CPUS="k8" DOCKER_TARGETS="tools" docker-build
```

When finished (less than 2 minutes), the binary tools are output in
`libcoral/out/k8/tools/`.

If you already have the repo and did not include submodules,
you can add them with this command from the repo root:

```
git submodule init && git submodule update
```

You can also build all targets for all supported CPUs with this:

```
bash scripts/build.sh
```

The following is a summary of each tool.


## `append_recurrent_links`

Creates recurrent networks for the Edge TPU with a hidden
saved state. *Without this tool* (and when not using the TF LSTM op), creating
your own recurrent network that can compile for the Edge TPU requires that your
model output the saved state, and then your application must pass the saved
state back into your model with each iteration. So by instead passing such a
model (already compiled for the Edge TPU) to `append_recurrent_links`, you can
make that saved state hidden again so your application code can focus on the
final output.

To create an RNN with this tool, you first need to cut the recurrent node so the
saved state is actually output at the end. Then pass the model to this tool and
specify the names for the saved state tensors to the `input_tensor_names` and
`output_tensor_names` arguments, which must have a one-to-one mapping, and in
order. For example:

```
./append_recurrent_links \
  --input_graph test_data/tools/split_concat_edgetpu.tflite \
  --output_graph /tmp/rnn_output_edgetpu.tflite \
  --input_tensor_names inputs/rnn1,inputs/rnn2 \
  --output_tensor_names outputs/rnn1,outputs/rnn2
```

For detail on all flags, run `./append_recurrent_links --help`.


## `join_tflite_models`

Concatenates two `.tflite` model files. It may be used for a
variety of tasks where you want one model to feed into another (assuming the
output tensor of the first model matches the input tensor of the second model),
but it is primarily intended to create models compatible with the
[`ImprintingEngine`](https://coral.ai/docs/reference/py/pycoral.learn.imprinting/)
API.

For example:

```
./join_tflite_models \
  --input_graph_base=mobilenet_v1_embedding_extractor_edgetpu.tflite \
  --input_graph_head=mobilenet_v1_last_layers.tflite \
  --output_graph=mobilenet_v1_l2norm_quant_edgetpu.tflite
```

For detail on all flags, run `./join_tflite_models --help`.


## `split_fc`

This is designed to pre-process `.tflite` files with a fully-connected
layer that's too big for the Edge TPU Compiler. For example, if you have a
model with a huge fully-connected layer, the Edge TPU Compiler previously might
have cut that layer from the Edge TPU delegate and instead execute it on the
CPU, due to the size of the weights applied to that layer. So the `split_fc`
tool divides that layer's weights matrix into smaller blocks using block-wise
matrix multiplication (you can control the ratio of the split operation). The
split_fc tool outputs a new `.tflite` file that you can pass to the Edge TPU
Compiler and the compiled output will then include the fully-connected layer in
the Edge TPU delegate.


## `partition_with_profiling`

Compiles your model into segments you can use for model pipelining,
using a segmentation strategy that optimizes the overall pipeline throughput
(it reduces the difference in latency between each segment).

For details, see the [Profiling-based partitioner
README](partitioner#profiling-based-partitioner-for-the-edge-tpu-compiler).


## `model_pipelining_performance_analysis`

Analyzes your model pipelining performance. Pass it the directory where
your segments are saved, the base filename shared by them all, and the number
of segments, and it then runs the pipeline and prints the latencies for each
segment.

For detail on all flags, run `./model_pipelining_performance_analysis --help`.


## `multiple_tpus_performance_analysis`

Analyzes the performance of your multi-Edge-TPU system. Instead of taking a list
of models to test, it uses a collection of 8 known models and runs a specified
number of inferences using all possible Edge TPU combinations, and then reports
the runtimes for each.

It accepts just one flag, `--num_requests`, which is the number of inferences
you want to run (default is 30000).

