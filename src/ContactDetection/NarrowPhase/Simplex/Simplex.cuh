//
// Created by iqraa on 11-3-25.
//

#ifndef SIMPLEX_CUH
#define SIMPLEX_CUH

#include <vector>

class Simplex {
public:
    Simplex()
    {
        points.reserve(4); // Maximum of 4 points in 3D
    }

    void add(const float3 &point)
    {
        points.push_back(point);
    }

    void clear()
    {
        points.clear();
    }

    size_t size() const
    {
        return points.size();
    }

    const float3 &operator[](size_t index) const
    {
        return points[index];
    }

    float3 &operator[](size_t index)
    {
        return points[index];
    }

    void removeLastPoint()
    {
        if (!points.empty())
        {
            points.pop_back();
        }
    }

    // Sets simplex to the given points
    void set(const std::vector<float3> &newPoints) {
        points = newPoints;
    }

    std::vector<float3> &getPoints() {
        return points;
    }

private:
    std::vector<float3> points;
};


#endif //SIMPLEX_CUH
