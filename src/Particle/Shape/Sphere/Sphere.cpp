//
// Created by iqraa on 14-2-25.
//

#include "Sphere.hpp"


Sphere::Sphere(float radius): radius_(radius) {
    setShapeType(SPHERE);
}

Sphere::Sphere(const Sphere& sphere) {
    radius_ = sphere.radius_;
    setShapeType(SPHERE);
};


void Sphere::setRadius(float radius) {radius_ = radius;}
