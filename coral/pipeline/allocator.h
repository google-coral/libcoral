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

#ifndef LIBCORAL_CORAL_PIPELINE_ALLOCATOR_H_
#define LIBCORAL_CORAL_PIPELINE_ALLOCATOR_H_

#include <cstddef>
#include <cstdlib>

namespace coral {

class Buffer {
 public:
  Buffer() = default;
  virtual ~Buffer() = default;

  Buffer(const Buffer& buffer) = delete;
  Buffer& operator=(const Buffer& buffer) = delete;

  // Returns user space pointer. Returned pointer can be nullptr.
  virtual void* ptr() = 0;

  // Maps buffer to host address space and returns the pointer. Returns nullptr
  // if the mapping was not successful. If the buffer was mapped before or does
  // not require mapping, then this is an no-op.
  //
  // Note if the underlying buffer is backed by DMA, DMA_BUF_IOCTL_SYNC ioctl
  // call might be needed for cache coherence. When such memory is consumed by
  // Edge TPU, such ioctl call is NOT necessary as the driver only uses the
  // pointer to identify the backing physical pages for DMA.
  virtual void* MapToHost() { return nullptr; }

  // Unmaps buffer from host address space. Returns true if successful; false
  // otherwise. This should be called explicitly if `MapToHost()` was called
  // before. Calling on an already unmapped buffer is a no-op.
  virtual bool UnmapFromHost() { return false; }

  // Returns file descriptor, -1 if buffer is NOT backed by file descriptor.
  virtual int fd() { return -1; }
};

class Allocator {
 public:
  Allocator() = default;
  virtual ~Allocator() = default;

  Allocator(const Allocator&) = delete;
  Allocator& operator=(const Allocator&) = delete;

  // Allocates `size_bytes` bytes of memory.
  // @param size_bytes The number of bytes to allocate.
  // @return A pointer to valid Buffer object.
  virtual Buffer* Alloc(size_t size_bytes) = 0;

  // Deallocates memory at the given block.
  virtual void Free(Buffer* buffer) = 0;
};
}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_ALLOCATOR_H_
