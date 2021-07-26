# libcoral

This repository contains sources for the libcoral C++ API, which provides
convenient functions to perform inferencing and on-device transfer learning
with TensorFlow Lite models on [Coral devices](https://coral.ai/products/).

For developer documentation, see our guide to [Run inference on the Edge TPU
with C++](https://coral.ai/docs/edgetpu/tflite-cpp/) and check out the
[libcoral API reference](https://coral.ai/docs/reference/cpp/).

## Compilation

Be sure to clone this repo with submodules:

```
git clone --recurse-submodules https://github.com/google-coral/libcoral
```

If you already cloned without the submodules. You can add them with this:

```
cd libcoral

git submodule init && git submodule update
```

Then you can build everything using `make` command which invokes
[Bazel](https://bazel.build/) internally.

For example, run `make tests` to build all C++ unit tests or `make benchmarks`
to build all C++ benchmarks. To get the list of all available make targets run
`make help`. All output goes to `out` directory.

### Linux

On Linux you can compile natively or cross-compile for 32-bit and 64-bit ARM
CPUs.

To compile natively you need to install at least the following packages:

```
sudo apt-get install -y build-essential \
                        libpython3-dev \
                        libusb-1.0-0-dev \
```

and to cross-compile:

```
sudo dpkg --add-architecture armhf
sudo apt-get install -y crossbuild-essential-armhf \
                        libpython3-dev:armhf \
                        libusb-1.0-0-dev:armhf

sudo dpkg --add-architecture arm64
sudo apt-get install -y crossbuild-essential-arm64 \
                        libpython3-dev:arm64 \
                        libusb-1.0-0-dev:arm64
```

Compilation or cross-compilation is done by setting CPU variable for `make`
command:

```
make CPU=k8      tests  # Builds for x86_64 (default CPU value)
make CPU=armv7a  tests  # Builds for ARMv7-A, e.g. Pi 3 or Pi 4
make CPU=aarch64 tests  # Builds for ARMv8, e.g. Coral Dev Board
```

### macOS

You need to install the following software:

1.  Xcode from https://developer.apple.com/xcode/
1.  Xcode Command Line Tools: `xcode-select --install`
1.  Bazel for macOS from https://github.com/bazelbuild/bazel/releases
1.  MacPorts from https://www.macports.org/install.php
1.  Ports of `python` interpreter and `numpy` library: `sudo port install
    python35 python36 python37 py35-numpy py36-numpy py37-numpy`
1.  Port of `libusb` library: `sudo port install libusb`

Right after that all normal `make` commands should work as usual. You can run
`make tests` to compile all C++ unit tests natively on macOS.

### Docker

Docker allows to avoid complicated environment setup and build binaries for
Linux on other operating systems without complicated setup, e.g.,

```
make DOCKER_IMAGE=debian:buster DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build
make DOCKER_IMAGE=ubuntu:18.04 DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build
```
