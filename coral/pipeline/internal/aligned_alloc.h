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

#ifndef LIBCORAL_CORAL_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_
#define LIBCORAL_CORAL_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_

#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace coral {
namespace internal {

inline void* AlignedAlloc(int alignment, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* ptr;
  if (posix_memalign(&ptr, alignment, size) == 0) return ptr;
  return nullptr;
#endif
}

inline void AlignedFree(void* aligned_memory) {
#if defined(_WIN32)
  _aligned_free(aligned_memory);
#else
  free(aligned_memory);
#endif
}

}  // namespace internal
}  // namespace coral

#endif  // LIBCORAL_CORAL_PIPELINE_INTERNAL_ALIGNED_ALLOC_H_
