/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "coral/tools/partitioner/profiling_based_partitioner_ondevice_lib.h"
#include "gtest/gtest.h"

namespace coral {
namespace {

TEST(ProfilingBasedPartitionerOndeviceTest, Tf1InceptionV3) {
  TestProfilingBasedPartitioner("inception_v3_299_quant",
                                /*num_segments=*/2,
                                /*expected_speedup=*/2,
                                /*partition_search_step=*/16);
}

TEST(ProfilingBasedPartitionerOndeviceTest, Tf1InceptionV4) {
  TestProfilingBasedPartitioner("inception_v4_299_quant",
                                /*num_segments=*/4,
                                /*expected_speedup=*/4.5,
                                /*partition_search_step=*/16);
}

TEST(ProfilingBasedPartitionerOndeviceTest, Tf2ResnetV150) {
  TestProfilingBasedPartitioner("tfhub_tf2_resnet_50_imagenet_ptq",
                                /*num_segments=*/3,
                                /*expected_speedup=*/6,
                                /*partition_search_step=*/1);
}

TEST(ProfilingBasedPartitionerOndeviceTest, Tf2SsdMobilenetV1Fpn) {
  TestProfilingBasedPartitioner("tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq",
                                /*num_segments=*/4,
                                /*expected_speedup=*/2.5,
                                /*partition_search_step=*/8);
}

TEST(ProfilingBasedPartitionerOndeviceTest, EfficientDetLite1_384) {
  TestProfilingBasedPartitioner("efficientdet_lite1_384_ptq",
                                /*num_segments=*/2,
                                /*expected_speedup=*/1.3,
                                /*partition_search_step=*/16,
                                /*delegate_search_step=*/4,
                                /*diff_threshold_ns=*/1000000,
                                /*initial_bounds=*/{30000000, 40000000});
}

TEST(ProfilingBasedPartitionerOndeviceTest, EfficientDetLite2_448) {
  TestProfilingBasedPartitioner("efficientdet_lite2_448_ptq",
                                /*num_segments=*/3,
                                /*expected_speedup=*/1.6,
                                /*partition_search_step=*/8,
                                /*delegate_search_step=*/4,
                                /*diff_threshold_ns=*/1000000,
                                /*initial_bounds=*/{30000000, 40000000});
}

TEST(ProfilingBasedPartitionerOndeviceTest, EfficientDetLite3_512) {
  TestProfilingBasedPartitioner("efficientdet_lite3_512_ptq",
                                /*num_segments=*/4,
                                /*expected_speedup=*/1.9,
                                /*partition_search_step=*/8,
                                /*delegate_search_step=*/4,
                                /*diff_threshold_ns=*/1000000,
                                /*initial_bounds=*/{30000000, 50000000});
}

TEST(ProfilingBasedPartitionerOndeviceTest, EfficientDetLite3x_640) {
  TestProfilingBasedPartitioner("efficientdet_lite3x_640_ptq",
                                /*num_segments=*/4,
                                /*expected_speedup=*/2.5,
                                /*partition_search_step=*/8,
                                /*delegate_search_step=*/4,
                                /*diff_threshold_ns=*/1000000,
                                /*initial_bounds=*/{30000000, 60000000});
}

}  // namespace
}  // namespace coral
