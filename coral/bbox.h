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

#ifndef LIBCORAL_CORAL_BBOX_H_
#define LIBCORAL_CORAL_BBOX_H_

#include <algorithm>
#include <ostream>
#include <string>

#include "absl/strings/substitute.h"
#include "glog/logging.h"

namespace coral {

// Represents the bounding box of a detected object.
template <typename T>
struct BBox {
  // Creates a `BBox` (box-corner encoding) from centroid box encodings.
  // @param center_y Box center y-coordinate.
  // @param center_x Box center x-coordinate.
  // @param height Box height.
  // @param width Box width.
  // @returns A `BBox` instance.
  static BBox<T> FromCenterSize(T center_y, T center_x, T height, T width) {
    const auto half_height = height / 2;
    const auto half_width = width / 2;
    return BBox<T>{center_y - half_height,  // ymin
                   center_x - half_width,   // xmin
                   center_y + half_height,  // ymax
                   center_x + half_width};  // xmax
  }

  // The box y-minimum (top-most) point.
  T ymin;
  // The box x-minimum (left-most) point.
  T xmin;
  // The box y-maximum (bottom-most) point.
  T ymax;
  // The box x-maximum (right-most) point.
  T xmax;

  // Gets the box width.
  T width() const { return xmax - xmin; }
  // Gets the box height.
  T height() const { return ymax - ymin; }
  // Gets the box area.
  T area() const { return width() * height(); }
  // Checks whether the box is a valid rectangle (width >= 0 and height >= 0).
  bool valid() const { return xmin <= xmax && ymin <= ymax; }
};

template <typename T>
bool operator==(const BBox<T>& a, const BBox<T>& b) {
  return a.ymin == b.ymin &&  //
         a.xmin == b.xmin &&  //
         a.ymax == b.ymax &&  //
         a.xmax == b.xmax;
}

template <typename T>
bool operator!=(const BBox<T>& a, const BBox<T>& b) {
  return !(a == b);
}

template <typename T>
std::string ToString(const BBox<T>& bbox) {
  return absl::Substitute("BBox(ymin=$0,xmin=$1,ymax=$2,xmax=$3)", bbox.ymin,
                          bbox.xmin, bbox.ymax, bbox.xmax);
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const BBox<T>& bbox) {
  return stream << ToString(bbox);
}

// Gets a `BBox` representing the intersection between two given boxes.
template <typename T>
BBox<T> Intersection(const BBox<T>& a, const BBox<T>& b) {
  return {std::max(a.ymin, b.ymin),  //
          std::max(a.xmin, b.xmin),  //
          std::min(a.ymax, b.ymax),  //
          std::min(a.xmax, b.xmax)};
}

// Gets a `BBox` representing the union of two given boxes.
template <typename T>
BBox<T> Union(const BBox<T>& a, const BBox<T>& b) {
  return {std::min(a.ymin, b.ymin),  //
          std::min(a.xmin, b.xmin),  //
          std::max(a.ymax, b.ymax),  //
          std::max(a.xmax, b.xmax)};
}

// Gets the intersection-over-union value for two boxes.
template <typename T>
float IntersectionOverUnion(const BBox<T>& a, const BBox<T>& b) {
  CHECK(a.valid());
  CHECK(b.valid());
  const auto intersection = Intersection(a, b);
  if (!intersection.valid()) return T(0);
  const auto common_area = intersection.area();
  return common_area / (a.area() + b.area() - common_area);
}

}  // namespace coral

#endif  // LIBCORAL_CORAL_BBOX_H_
