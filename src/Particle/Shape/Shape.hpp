//
// Created by iqraa on 14-2-25.
//

#ifndef SHAPE_H
#define SHAPE_H
#include <cuda_runtime_api.h>


class Shape
{
public:
  Shape();

  virtual ~Shape();

  __host__ __device__
  int getId()  const { return id_; }

  __host__ __device__
  void setId(const int id) { id_ = id; }

private:
  int id_ = 0;
};

#endif //SHAPE_H
