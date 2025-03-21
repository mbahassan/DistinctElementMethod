//
// Created by iqraa on 20-3-25.
//

#ifndef POSITION_H
#define POSITION_H

#include "Tools/ArthmiticOperator/MathOperators.hpp"
template<typename TYPE>
class Position {
public:

    Position(TYPE* p) : position_{0.f, 0.f, 0.f}, owner(p) {}

    // This gets called when you do: position = {x, y, z}
    Position& operator=(const float3& newPos) {
        float3 translation = newPos - position_;
        position_ = newPos;

        // Call a method to update all geometry
        if (owner) {
            owner->update(translation);
        }

        return *this;
    }

    // This lets you use position as if it were a float3
    operator float3() const { return position_; }
private:
    float3 position_;

    TYPE* owner;
};



#endif //POSITION_H
