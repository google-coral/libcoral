#include "coral/pipeline/internal/memory_pool_allocator.h"

#include <memory>

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
      std::unordered_map<size_t, int>({{1024, 2}}));
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
      std::unordered_map<size_t, int>({{1024, 1}}));
  auto* first_buffer = allocator->Alloc(512);
  EXPECT_EQ(first_buffer->ptr(), nullptr);

  allocator->Free(first_buffer);
}

TEST(MemoryPoolAllocator, Alignment) {
  auto allocator = absl::make_unique<MemoryPoolAllocator>(
      std::unordered_map<size_t, int>({{1023, 2}}));
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
      std::unordered_map<size_t, int>({{512, 1}}));
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
