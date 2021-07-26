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

#include "coral/pipeline/internal/memory_pool_allocator.h"

#include <cstdlib>

#include "coral/pipeline/internal/aligned_alloc.h"

namespace coral {
namespace internal {
namespace {
int RoundUp(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}
}  // namespace

MemoryPoolAllocator::MemoryPoolAllocator(
    const absl::flat_hash_map<size_t, int>& size_to_copy_map) {
  // Calculate total number of bytes needed.
  // Each block size is rounded up to the nearest number that's a multiple of
  // kAlignment bytes.
  size_t total_size_bytes = 0;
  std::vector<int> block_sizes, aligned_block_sizes, copies_per_size;
  const int num_unique_blocks = size_to_copy_map.size();
  block_sizes.reserve(num_unique_blocks);
  aligned_block_sizes.reserve(num_unique_blocks);
  copies_per_size.reserve(num_unique_blocks);
  for (const auto& pair : size_to_copy_map) {
    block_sizes.push_back(pair.first);
    copies_per_size.push_back(pair.second);
    aligned_block_sizes.push_back(RoundUp(block_sizes.back(), kAlignment));
    total_size_bytes += aligned_block_sizes.back() * copies_per_size.back();
  }
  VLOG(1) << "Total pool size (bytes): " << total_size_bytes;

  pool_ = AlignedAlloc(kAlignment, total_size_bytes);
  CHECK(pool_) << "Not enough memory, requested: " << total_size_bytes;

  VLOG(1) << "Constructing memory blocks map...";
  auto tmp_addr = base_addr();
  for (int i = 0; i < num_unique_blocks; ++i) {
    memory_blocks_.emplace(
        std::piecewise_construct, std::forward_as_tuple(block_sizes[i]),
        std::forward_as_tuple(tmp_addr, aligned_block_sizes[i],
                              copies_per_size[i]));
    tmp_addr += aligned_block_sizes[i] * copies_per_size[i];
  }
}
}  // namespace internal
}  // namespace coral
