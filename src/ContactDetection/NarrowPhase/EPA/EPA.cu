//
// Created by iqraa on 12-3-25.
//

#include "EPA.cuh"

EPA::Triangle::Triangle(int a, int b, int c, const std::vector<float3> &vertices) {
    indices[0] = a;
    indices[1] = b;
    indices[2] = c;

    // Calculate face normal
    float3 ab = vertices[b] - vertices[a];
    float3 ac = vertices[c] - vertices[a];
    normal = ab.cross(ac);

    // Ensure normal points outward from origin
    float3 ao = vertices[a]; // Vector from origin to point A
    if (normal.dot(ao) < 0) {
        // Swap vertices to flip normal
        indices[1] = c;
        indices[2] = b;
        normal = -normal;
    }

    // Normalize the normal
    float length = normal.length();
    if (length > 0.0001f) {
        normal = normal * (1.0f / length);
    }

    // Calculate distance from origin
    distance = normal.dot(vertices[a]);
}

// Determine if a point is in front of this face
bool EPA::Triangle::isFrontFacing(const float3 &point, const std::vector<float3> &vertices) const {
    float3 v0 = vertices[indices[0]];
    return (point - v0).dot(normal) > 0;
}


EPA::Edge::Edge(int a, int b) : a(std::min(a, b)), b(std::max(a, b)) {
}

bool EPA::Edge::operator==(const Edge &other) const {
    return a == other.a && b == other.b;
}

bool EPA::Edge::operator<(const Edge &other) const {
    if (a != other.a) return a < other.a;
    return b < other.b;
}
