# Profiling-based partitioner for the Edge TPU Compiler

The `partition_with_profiling` tool segments an Edge TPU model for
[model pipelining](https://coral.ai/docs/edgetpu/pipeline/), using a
segmentation strategy that improves the overall throughput in pipelining.
(Whereas, the [Edge TPU Compiler's `--num_segments`
argument](https://coral.ai/docs/edgetpu/pipeline/#segment-a-model)
divides the model so each segment has a roughly the same amount of parameter
data.)

Because the overall throughput is usually constrained by the latency of the
slowest segment, the algorithm in this tool attempts to minimize the latency
difference between all segments. First, it segments the model using the Edge TPU
Compiler's `--num_segments` argument and measures the latency of each segment on
the Edge TPU. From there, it determines a "target" latency for all segments
(roughly the average of all the baseline segment latencies), it intelligently
re-segments the original model to distribute the execution time, and then it
measures the latency again. It repeats this until all segments have roughly the
same latency.

**Note:** Before using the profiling-based partitioner, we suggest you first try
segmenting your model as described in the guide to [pipeline a model with
multiple Edge TPUs](https://coral.ai/docs/edgetpu/pipeline/). Then, only if the
latency does not meet your demands, should you need to use this tool. That is
unless your model has a non-trivial amount of operations that execute on the CPU
and/or it has branches in the graph (such as in an SSD model)—in which case,
only the profiling-based partitioner will accurately segment the model based on
how much of the graph actually executes on the Edge TPU.

## Requirements

Because this strategy requires executing the model on the Edge TPU and
iteratively re-compiling the model, you must use `partition_with_profiling`
on a system that both has access to the number of Edge TPUs required by your
pipelined model **and** meets the [Edge TPU Compiler's system
requirements](https://coral.ai/docs/edgetpu/compiler/#system-requirements).

Basically, that means you need an x86-64 system with Debian 6 or higher
and multiple Edge TPUs.

You also need [Docker](https://docs.docker.com/get-docker/) to build the
partitioner.

**Note:** In order to accurately model your pipeline throughput, the Edge TPUs
used by the profiling-based partitioner should use the same interface (PCIe or
USB) as those in your production system, and the CPU should also be the same as
your production system. If that's not possible, that's okay, but beware that
there may be a difference between the throughput measured by the profiling-based
partitioner and the throughput on your production system—although the new
segmented model should still provide an improved throughput on any system.

## Build the partitioner

To compile the partitioner, you first need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
(optionally, but we recommend) [Docker](https://docs.docker.com/install/).

Then clone this repo with submodules and build the tools:

```
git clone --recurse-submodules https://github.com/google-coral/libcoral.git

cd libcoral

make DOCKER_IMAGE="ubuntu:18.04" DOCKER_CPUS="k8" DOCKER_TARGETS="tools" docker-build
```

When finished (less than 2 minutes), the tool is output at
`libcoral/out/k8/tools/partitioner/partition_with_profiling`.

If you already have the repo and did not include submodules,
you can add them with this command from the repo root:

```
git submodule init && git submodule update
```

You can also build all targets for all supported CPUs with this:

```
bash scripts/build.sh
```


## Run the partitioner

Find the location where you installed the
[Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/), make sure your
system has the number of Edge TPUs specified for `num_segments`, and then
run the partitioner as follows:

```
./partition_with_profiling \
  --edgetpu_compiler_binary $PATH_TO_COMPILER \
  --model_path $PATH_TO_MODEL \
  --output_dir $OUT_DIR \
  --num_segments $NUM_SEGMENTS
```

Running this tool takes longer than running the Edge TPU Compiler directly
because it may re-segment the model many times.

For detail on all flags, run `./partition_with_profiling --help` or see the
[documentation here](/docs/edgetpu/compiler/#profiling-partitioner).

