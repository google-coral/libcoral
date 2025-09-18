# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
OS := $(shell uname -s)

# Allowed CPU values: k8, armv7a, aarch64, darwin
ifeq ($(OS),Linux)
CPU ?= k8
TEST_FILTER ?= cat
else ifeq ($(OS),Darwin)
ifeq ($(ARCH),arm)
CPU ?= darwin_arm64
else
CPU ?= darwin_x86_64
endif
TEST_FILTER ?= grep -v dmabuf
else
$(error $(OS) is not supported)
endif

ifeq ($(filter $(CPU),k8 armv7a aarch64 darwin_arm64 darwin_x86_64),)
$(error CPU must be k8, armv7a, aarch64, or darwin_arm64 darwin_x86_64)
endif

# Allowed COMPILATION_MODE values: opt, dbg, fastbuild
COMPILATION_MODE ?= opt
ifeq ($(filter $(COMPILATION_MODE),opt dbg fastbuild),)
$(error COMPILATION_MODE must be opt, dbg, or fastbuild)
endif

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --compilation_mode=$(COMPILATION_MODE) \
                     --cpu=$(CPU)

ifeq ($(CPU),aarch64)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
else ifeq ($(CPU),armv7a)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off --copt=-mfp16-format=ieee
endif

# $(1): pattern, $(2) destination directory
define copy_out_files
pushd $(BAZEL_OUT_DIR); \
for f in `find . -name $(1) -type f`; do \
	mkdir -p $(2)/`dirname $$f`; \
	cp -f $(BAZEL_OUT_DIR)/$$f $(2)/$$f; \
done; \
popd
endef

EXAMPLES_OUT_DIR   := $(MAKEFILE_DIR)/out/$(CPU)/examples
TOOLS_OUT_DIR      := $(MAKEFILE_DIR)/out/$(CPU)/tools
TESTS_OUT_DIR      := $(MAKEFILE_DIR)/out/$(CPU)/tests
BENCHMARKS_OUT_DIR := $(MAKEFILE_DIR)/out/$(CPU)/benchmarks

.PHONY: all \
        tests \
        benchmarks \
        tools \
        examples \
        clean \
        help

all: tests benchmarks tools examples

tests:
	bazel build $(BAZEL_BUILD_FLAGS) $(shell bazel query 'kind(cc_.*test, //coral/...)' | $(TEST_FILTER))
	$(call copy_out_files,"*_test",$(TESTS_OUT_DIR))

benchmarks:
	bazel build $(BAZEL_BUILD_FLAGS) $(shell bazel query 'kind(cc_binary, //coral/...)' | grep benchmark)
	$(call copy_out_files,"*_benchmark",$(BENCHMARKS_OUT_DIR))

tools:
	bazel build $(BAZEL_BUILD_FLAGS) //coral/tools:append_recurrent_links \
	                                 //coral/tools:join_tflite_models \
	                                 //coral/tools:multiple_tpus_performance_analysis \
	                                 //coral/tools:model_pipelining_performance_analysis \
	                                 //coral/tools/partitioner:partition_with_profiling
	mkdir -p $(TOOLS_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/coral/tools/append_recurrent_links \
	      $(BAZEL_OUT_DIR)/coral/tools/join_tflite_models \
	      $(BAZEL_OUT_DIR)/coral/tools/multiple_tpus_performance_analysis \
	      $(BAZEL_OUT_DIR)/coral/tools/model_pipelining_performance_analysis \
	      $(TOOLS_OUT_DIR)
	mkdir -p $(TOOLS_OUT_DIR)/partitioner
	cp -f $(BAZEL_OUT_DIR)/coral/tools/partitioner/partition_with_profiling \
	      $(TOOLS_OUT_DIR)/partitioner

examples:
	bazel build $(BAZEL_BUILD_FLAGS) //coral/examples:two_models_one_tpu \
	                                 //coral/examples:two_models_two_tpus_threaded \
	                                 //coral/examples:model_pipelining \
	                                 //coral/examples:classify_image \
	                                 //coral/examples:backprop_last_layer
	mkdir -p $(EXAMPLES_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/coral/examples/two_models_one_tpu \
	      $(BAZEL_OUT_DIR)/coral/examples/two_models_two_tpus_threaded \
	      $(BAZEL_OUT_DIR)/coral/examples/model_pipelining \
	      $(BAZEL_OUT_DIR)/coral/examples/classify_image \
	      $(BAZEL_OUT_DIR)/coral/examples/backprop_last_layer \
	      $(EXAMPLES_OUT_DIR)

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/out

TEST_ENV := $(shell test -L $(MAKEFILE_DIR)/test_data && echo 1)
DOCKER_WORKSPACE := $(MAKEFILE_DIR)/$(if $(TEST_ENV),..,)
DOCKER_WORKSPACE_CD := $(if $(TEST_ENV),libcoral,)
DOCKER_CPUS := k8 armv7a aarch64
DOCKER_TAG_BASE := coral-edgetpu
include $(MAKEFILE_DIR)/docker/docker.mk

help:
	@echo "make all        - Build all C++ code"
	@echo "make tests      - Build all C++ tests"
	@echo "make benchmarks - Build all C++ benchmarks"
	@echo "make tools      - Build all C++ tools"
	@echo "make examples   - Build all C++ examples"
	@echo "make clean      - Remove generated files"
	@echo "make help       - Print help message"
