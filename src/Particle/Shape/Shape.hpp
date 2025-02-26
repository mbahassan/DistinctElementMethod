//
// Created by iqraa on 14-2-25.
//

#ifndef SHAPE_H
#define SHAPE_H
#include <cuda_runtime_api.h>
#include <memory>


class Shape
{
public:
  enum ShapeType {SPHERE, CYLINDER, CUBE};

  Shape() = default;

  ~Shape()= default;

  __host__ __device__
  void setId(const unsigned int id) { id_ = id; }

  __host__ __device__
  void setShapeType(const ShapeType shapeType){ shapeType_ = shapeType; }

  __host__ __device__
  int getId()  const { return id_; }

  __host__ __device__
  ShapeType getShapeType() const {return shapeType_;}

  /// Sphere
  virtual float getRadius() const = 0;

private:
  unsigned int id_ = 0;

  // Default Shape is Sphere
  ShapeType shapeType_ = SPHERE;
};

#endif //SHAPE_H
