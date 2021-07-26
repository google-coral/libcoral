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

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace coral {
namespace internal {
namespace {

inline uintptr_t addr(Buffer* buffer) {
  return reinterpret_cast<uintptr_t>(buffer->ptr());
}

TEST(MemoryPoolAllocator, Allocate) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      absl::flat_hash_map<size_t, int>({{1024, 2}}));
  auto* first_buffer = allocator->Alloc(1024);
  EXPECT_GE(addr(first_buffer), allocator->base_addr());
  auto* second_buffer = allocator->Alloc(1024);
  EXPECT_GE(addr(second_buffer), allocator->base_addr());
  EXPECT_NE(addr(first_buffer), addr(second_buffer));
  auto* third_buffer = allocator->Alloc(1024);
  EXPECT_EQ(third_buffer->ptr(), nullptr);

  allocator->Free(first_buffer);
  allocator->Free(second_buffer);
  allocator->Free(third_buffer);
}

TEST(MemoryPoolAllocator, AllocateUnsupportedSize) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      absl::flat_hash_map<size_t, int>({{1024, 1}}));
  auto* first_buffer = allocator->Alloc(512);
  EXPECT_EQ(first_buffer->ptr(), nullptr);

  allocator->Free(first_buffer);
}

TEST(MemoryPoolAllocator, Alignment) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      absl::flat_hash_map<size_t, int>({{1023, 2}}));
  auto* first_buffer = allocator->Alloc(1023);
  EXPECT_EQ(addr(first_buffer) % MemoryPoolAllocator::kAlignment, 0);
  EXPECT_GE(addr(first_buffer), allocator->base_addr());
  auto* second_buffer = allocator->Alloc(1023);
  EXPECT_EQ(addr(first_buffer) % MemoryPoolAllocator::kAlignment, 0);
  EXPECT_GE(addr(second_buffer), allocator->base_addr());
  EXPECT_NE(addr(first_buffer), addr(second_buffer));

  allocator->Free(first_buffer);
  allocator->Free(second_buffer);
}

TEST(MemoryPoolAllocator, Deallocate) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      absl::flat_hash_map<size_t, int>({{512, 1}}));
  auto* first_buffer = allocator->Alloc(512);
  EXPECT_EQ(addr(first_buffer), allocator->base_addr());
  auto* second_buffer = allocator->Alloc(512);
  EXPECT_EQ(second_buffer->ptr(), nullptr);
  allocator->Free(first_buffer);
  auto* third_buffer = allocator->Alloc(512);
  EXPECT_EQ(addr(third_buffer), allocator->base_addr());

  allocator->Free(second_buffer);
  allocator->Free(third_buffer);
}

}  // namespace
}  // namespace internal
}  // namespace coral
