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

#ifndef LIBCORAL_CORAL_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_
#define LIBCORAL_CORAL_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_

#include <cstdint>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "coral/pipeline/allocator.h"
#include "coral/pipeline/internal/aligned_alloc.h"
#include "glog/logging.h"

namespace coral {
namespace internal {

class PoolBuffer : public Buffer {
 public:
  explicit PoolBuffer(void* ptr, size_t size_bytes)
      : ptr_(ptr), size_bytes_(size_bytes) {}

  void* ptr() override { return ptr_; }

  size_t size_bytes() { return size_bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t size_bytes_ = 0;
};

// MemoryPoolAllocator can only allocate memory of predefined sizes, with each
// predefined size allocated at most K copies. For example,
//
//   auto allocator = MemoryPoolAllocator({{1024, 4}, {512, 8}});
//
// It defines an allocator that can allocate at most 4 copies of memory of 1024
// bytes, and at most 8 copies of memory of 512 bytes.
//
// This class is thread-safe.
class MemoryPoolAllocator : public Allocator {
 public:
  // Allocated addresses are kAlignment-byte-aligned.
  static constexpr int kAlignment = 8;

  // Key is memory block size, value is number of copies.
  explicit MemoryPoolAllocator(
      const absl::flat_hash_map<size_t, int>& size_to_copy_map);

  ~MemoryPoolAllocator() override {
    if (pool_) {
      AlignedFree(pool_);
    }
  }

  // Returned buffer has ptr()==nullptr if allocation fails, either
  // because the allocator cannot handle given `size_bytes`, or allocator runs
  // out of copies of `size_bytes`.
  Buffer* Alloc(size_t size_bytes) override {
    void* ptr = nullptr;
    auto it = memory_blocks_.find(size_bytes);
    if (it != memory_blocks_.end()) {
      ptr = it->second.get();
    }
    return new PoolBuffer(ptr, size_bytes);
  }

  void Free(Buffer* buffer) override {
    if (buffer) {
      if (buffer->ptr()) {
        auto size_bytes = static_cast<PoolBuffer*>(buffer)->size_bytes();
        auto it = memory_blocks_.find(size_bytes);
        if (it != memory_blocks_.end()) {
          it->second.release(buffer->ptr());
        }
      }
      delete buffer;
    }
  }

  // Returns base address of underlying memory pool.
  uintptr_t base_addr() const { return reinterpret_cast<uintptr_t>(pool_); }

 private:
  // Defines `num_copies` copies of memory blocks of size `block_size`.
  class MemoryBlocks {
   public:
    MemoryBlocks(uintptr_t base_addr, size_t block_size, int num_copies) {
      for (int i = 0; i < num_copies; ++i) {
        blocks_.push(reinterpret_cast<void*>(base_addr + i * block_size));
      }
    }

    // Returns next available block. Returns nullptr if none available.
    void* get() {
      absl::MutexLock lock(&mu_);
      if (blocks_.empty()) {
        return nullptr;
      } else {
        void* result = blocks_.front();
        blocks_.pop();
        return result;
      }
    }

    // Releases memory block.
    void release(void* p) {
      absl::MutexLock lock(&mu_);
      blocks_.push(p);
    }

   private:
    absl::Mutex mu_;
    std::queue<void*> blocks_ ABSL_GUARDED_BY(mu_);
  };

  // Key is predifined block size, value is the corresponding memory blocks.
  absl::node_hash_map<size_t, MemoryBlocks> memory_blocks_;

  // Underlying memory pool.
  void* pool_;
};

}  // namespace internal
}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_INTERNAL_MEMORY_POOL_ALLOCATOR_H_
