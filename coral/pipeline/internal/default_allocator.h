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

#ifndef LIBCORAL_CORAL_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
#define LIBCORAL_CORAL_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_

#include "coral/pipeline/allocator.h"

namespace coral {
namespace internal {

class HeapBuffer : public Buffer {
 public:
  explicit HeapBuffer(void *ptr) : ptr_(ptr) {}

  void *ptr() override { return ptr_; }

 private:
  void *ptr_ = nullptr;
};

class DefaultAllocator : public Allocator {
 public:
  DefaultAllocator() = default;
  ~DefaultAllocator() override = default;

  Buffer *Alloc(size_t size) override {
    return new HeapBuffer(std::malloc(size));
  }

  void Free(Buffer *buffer) override {
    if (buffer) {
      std::free(buffer->ptr());
      delete buffer;
    }
  }
};

}  // namespace internal
}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
