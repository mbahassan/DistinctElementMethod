//
// Created by iqraa on 14-2-25.
//

#include "Sphere.hpp"

Sphere::Sphere() {
    setShapeType(SPHERE);
}

Sphere::Sphere(float radius): radius_(radius) {
    setShapeType(SPHERE);
}

Sphere::Sphere(const Sphere& sphere) {
    radius_ = sphere.radius_;
    setShapeType(SPHERE);
};

Sphere::~Sphere()= default;

void Sphere::setRadius(float radius) {radius_ = radius;}

float Sphere::getRadius() const {return radius_;}

float Sphere::getVolume() const{return 4.0f/radius_*radius_*radius_/3.0f;}