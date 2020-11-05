#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_

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

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_DEFAULT_ALLOCATOR_H_
