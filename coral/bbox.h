#ifndef EDGETPU_CPP_BBOX_H_
#define EDGETPU_CPP_BBOX_H_

#include <algorithm>
#include <ostream>
#include <string>

#include "absl/strings/substitute.h"
#include "glog/logging.h"

namespace coral {

template <typename T>
struct BBox {
  static BBox<T> FromCenterSize(T center_y, T center_x, T height, T width) {
    const auto half_height = height / 2;
    const auto half_width = width / 2;
    return BBox<T>{center_y - half_height,  // ymin
                   center_x - half_width,   // xmin
                   center_y + half_height,  // ymax
                   center_x + half_width};  // xmax
  }

  T ymin;
  T xmin;
  T ymax;
  T xmax;

  T width() const { return xmax - xmin; }
  T height() const { return ymax - ymin; }
  T area() const { return width() * height(); }
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

template <typename T>
BBox<T> Intersection(const BBox<T>& a, const BBox<T>& b) {
  return {std::max(a.ymin, b.ymin),  //
          std::max(a.xmin, b.xmin),  //
          std::min(a.ymax, b.ymax),  //
          std::min(a.xmax, b.xmax)};
}

template <typename T>
BBox<T> Union(const BBox<T>& a, const BBox<T>& b) {
  return {std::min(a.ymin, b.ymin),  //
          std::min(a.xmin, b.xmin),  //
          std::max(a.ymax, b.ymax),  //
          std::max(a.xmax, b.xmax)};
}

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

#endif  // EDGETPU_CPP_BBOX_H_
