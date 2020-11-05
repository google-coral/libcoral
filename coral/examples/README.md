# C++ examples using Edge TPU

To build all the examples in this directory, you first need to
[install Bazel](https://docs.bazel.build/versions/master/install.html) and
(optional but we recommend)
[Docker](https://docs.docker.com/install/).

Then navigate up to the root `libcoral` directory and run the following command:

```
make DOCKER_IMAGE=debian:stretch DOCKER_CPUS="aarch64" DOCKER_TARGETS="examples" docker-build
```

When done, you'll find the example binaries in `libcoral/out/aarch64/examples/`.

The above command builds for `aarch64` (compatible with the Coral Dev Board),
but alternative CPU options are `k8`, `armv7a`, and `darwin`.

**Tip:** Instead of building on your computer, just
[run this Colab notebook](https://colab.sandbox.google.com/github/google-coral/tutorials/blob/master/build_cpp_examples.ipynb)
to build the examples and download the binaries.
