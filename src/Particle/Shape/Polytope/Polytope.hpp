//
// Created by iqraa on 13-3-25.
//

#ifndef POLYTOPE_H
#define POLYTOPE_H
#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>
#include <Tools/tinyobjloader/objloader.h>
#include <thrust/host_vector.h>
#include "Tools/quaternion/quaternion.hpp"



class Polytope : public Shape {
public:
    Polytope() {
        setShapeType(POLYHEDRAL);
    };

    explicit Polytope(const std::string &file) {
        setShapeType(POLYHEDRAL);

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Load the OBJ file
        bool ret = LoadObj(&attrib, &shapes, &materials, &warn, &err, file.c_str());

        if (!warn.empty()) std::cout << "WARN: " << warn << std::endl;
        if (!err.empty()) std::cerr << "ERR: " << err << std::endl;

        if (!ret) {
            std::cerr << "Failed to load .obj file." << std::endl;
            return;
        }

        // Fill vertices (each vertex has 3 consecutive floats)
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            float3 v;
            v.x = attrib.vertices[i];
            v.y = attrib.vertices[i + 1];
            v.z = attrib.vertices[i + 2];
            vertices.push_back(v);
            if (i == 0) {
                minPt = v;
                maxPt = v;
            } else {
                // Update bounding box
                minPt.x = std::min(minPt.x, v.x);
                minPt.y = std::min(minPt.y, v.y);
                minPt.z = std::min(minPt.z, v.z);

                maxPt.x = std::max(maxPt.x, v.x);
                maxPt.y = std::max(maxPt.y, v.y);
                maxPt.z = std::max(maxPt.z, v.z);
            }
        }

        for (size_t s = 0; s < shapes.size(); s++) {
            size_t index_offset = 0;
            // Iterate over the number of faces in the shape
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                int fv = shapes[s].mesh.num_face_vertices[f]; // number of vertices for face f
                Face face;
                // For each vertex in the face, get its index and add to face.indices
                for (size_t vi = 0; vi < fv; vi++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + vi];
                    face.indices.push_back(idx.vertex_index);
                }
                faces.push_back(face);
                index_offset += fv; // move to the next face's indices
            }
        }

        computeVolumeAndCoM(vertices, faces, volume, centerOfMass);
    }

    ~Polytope() override = default;

    float3 supportMapping(const float3 &direction) const {
        float maxDistance = vertices[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < vertices.size(); i += 1) {
            float distance = vertices[i] & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }
        return vertices[maxIdx];
    }


    __host__ __device__
    float getVolume() override { return volume; }

    __host__ __device__
    float3 getMin() override { return minPt; }

    __host__ __device__
    float3 getMax() override { return maxPt; }

    __host__ __device__
    float3 getCenterOfMass() const { return centerOfMass; }

    int getVerticesCount() const { return vertices.size(); }

    int getFacesCount() const { return faces.size(); }

    Face getFace(int i) const { return faces[i]; }

    float3 getVertex(int i) const { return vertices[i]; }

    // Method to get the orientation
    __host__ __device__
    Quaternion getOrientation() const {
        return orientation;
    }

private:
    // Compute volume and center of mass
    void computeVolumeAndCoM(
        const std::vector<float3> &vertices,
        const std::vector<Face> &faces,
        float &totalVolume,
        float3 &centerOfMass) {
        // Compute reference point r (the average of all vertices)
        float3 r = {0.f, 0.f, 0.f};
        for (const auto &v: vertices) {
            r = r + v;
        }
        r = r * (1.0f / vertices.size());

        totalVolume = 0.0f;
        centerOfMass = {0.f, 0.f, 0.f};

        // Loop over all faces
        for (const auto &face: faces) {
            // Decompose the face into triangles using the "fan" method.
            // Assumes the face is convex and planar.
            for (size_t i = 1; i + 1 < face.indices.size(); ++i) {
                float3 v0 = vertices[face.indices[0]];
                float3 v1 = vertices[face.indices[i]];
                float3 v2 = vertices[face.indices[i + 1]];

                // Compute differences relative to the reference point
                float3 d0 = v0 - r;
                float3 d1 = v1 - r;
                float3 d2 = v2 - r;

                // Compute volume of the tetrahedron (r, v0, v1, v2)
                double vol = (d0 & (d1 ^ d2)) / 6.0; // assuming '&' is dot and '^' is cross

                // Compute centroid of tetrahedron
                float3 tetCentroid = (r + v0 + v1 + v2) * (1.0f / 4.0f);

                centerOfMass = centerOfMass + tetCentroid * vol;
                totalVolume += vol;
            }
        }

        if (std::abs(totalVolume) > 1e-12)
            centerOfMass = centerOfMass * (1.0f / totalVolume);
        else
            std::cerr << "Warning: Total volume is near zero. Possibly open or invalid mesh." << std::endl;

        totalVolume = std::abs(totalVolume); // Use absolute volume
    }

    // Add these helper methods for eigenvalue/eigenvector computation
    void computeEigenvaluesAndEigenvectors(const Matrix3x3 &matrix, float eigenvalues[3], Matrix3x3 &eigenvectors) {
        // This is a simplified implementation for a 3x3 symmetric matrix
        // For production code, consider a more robust method like Jacobi iteration

        // Copy the matrix for manipulation
        float A[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                A[i][j] = matrix.m[i][j];
            }
        }

        // Simple power iteration method for dominant eigenvalue/eigenvector
        // In practice, you should use a more robust method
        float vec[3] = {1.0f, 0.0f, 0.0f};
        normalizeVector(vec);

        // Find first eigenvector (dominant)
        powerIteration(A, vec, eigenvalues[0], 20);
        for (int i = 0; i < 3; i++) {
            eigenvectors.m[i][0] = vec[i];
        }

        // Deflate the matrix to find second eigenvector
        Matrix3x3 deflated;
        deflateMatrix(A, vec, eigenvalues[0], deflated);

        // Find second eigenvector
        float vec2[3] = {0.0f, 1.0f, 0.0f};
        makeOrthogonal(vec2, vec);
        normalizeVector(vec2);

        powerIteration(deflated.m, vec2, eigenvalues[1], 20);
        for (int i = 0; i < 3; i++) {
            eigenvectors.m[i][1] = vec2[i];
        }

        // Last eigenvector is cross product of first two
        float vec3[3];
        vec3[0] = vec[1] * vec2[2] - vec[2] * vec2[1];
        vec3[1] = vec[2] * vec2[0] - vec[0] * vec2[2];
        vec3[2] = vec[0] * vec2[1] - vec[1] * vec2[0];
        normalizeVector(vec3);

        eigenvalues[2] = matrixVectorProduct(A, vec3)[0] * vec3[0] +
                         matrixVectorProduct(A, vec3)[1] * vec3[1] +
                         matrixVectorProduct(A, vec3)[2] * vec3[2];

        for (int i = 0; i < 3; i++) {
            eigenvectors.m[i][2] = vec3[i];
        }

        // Ensure right-handed system
        if (eigenvectors.determinant() < 0) {
            for (int i = 0; i < 3; i++) {
                eigenvectors.m[i][2] = -eigenvectors.m[i][2];
            }
        }
    }

    void normalizeVector(float vec[3]) {
        float magnitude = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        if (magnitude > 1e-10f) {
            vec[0] /= magnitude;
            vec[1] /= magnitude;
            vec[2] /= magnitude;
        }
    }

    void makeOrthogonal(float vec[3], const float ref[3]) {
        float dot = vec[0] * ref[0] + vec[1] * ref[1] + vec[2] * ref[2];
        vec[0] -= dot * ref[0];
        vec[1] -= dot * ref[1];
        vec[2] -= dot * ref[2];
    }

    void powerIteration(float matrix[3][3], float vec[3], float &eigenvalue, int iterations) {
        for (int iter = 0; iter < iterations; iter++) {
            float result[3] = {0, 0, 0};

            // Matrix-vector multiplication
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[i] += matrix[i][j] * vec[j];
                }
            }

            // Normalize the resulting vector
            normalizeVector(result);

            // Copy result back to vec
            for (int i = 0; i < 3; i++) {
                vec[i] = result[i];
            }
        }

        // Calculate the eigenvalue using the Rayleigh quotient
        float *result = matrixVectorProduct(matrix, vec);
        eigenvalue = vec[0] * result[0] + vec[1] * result[1] + vec[2] * result[2];
        delete[] result;
    }

    float *matrixVectorProduct(float matrix[3][3], const float vec[3]) {
        float *result = new float[3];
        for (int i = 0; i < 3; i++) {
            result[i] = 0;
            for (int j = 0; j < 3; j++) {
                result[i] += matrix[i][j] * vec[j];
            }
        }
        return result;
    }

    void deflateMatrix(float matrix[3][3], const float vec[3], float eigenvalue, Matrix3x3 &deflated) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                deflated.m[i][j] = matrix[i][j] - eigenvalue * vec[i] * vec[j];
            }
        }
    }

    // Convert rotation matrix to quaternion
    Quaternion matrixToQuaternion(const Matrix3x3 &m) {
        Quaternion q;
        float trace = m.m[0][0] + m.m[1][1] + m.m[2][2];

        if (trace > 0) {
            float s = 0.5f / sqrt(trace + 1.0f);
            q.w = 0.25f / s;
            q.x = (m.m[2][1] - m.m[1][2]) * s;
            q.y = (m.m[0][2] - m.m[2][0]) * s;
            q.z = (m.m[1][0] - m.m[0][1]) * s;
        } else {
            if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
                float s = 2.0f * sqrt(1.0f + m.m[0][0] - m.m[1][1] - m.m[2][2]);
                q.w = (m.m[2][1] - m.m[1][2]) / s;
                q.x = 0.25f * s;
                q.y = (m.m[0][1] + m.m[1][0]) / s;
                q.z = (m.m[0][2] + m.m[2][0]) / s;
            } else if (m.m[1][1] > m.m[2][2]) {
                float s = 2.0f * sqrt(1.0f + m.m[1][1] - m.m[0][0] - m.m[2][2]);
                q.w = (m.m[0][2] - m.m[2][0]) / s;
                q.x = (m.m[0][1] + m.m[1][0]) / s;
                q.y = 0.25f * s;
                q.z = (m.m[1][2] + m.m[2][1]) / s;
            } else {
                float s = 2.0f * sqrt(1.0f + m.m[2][2] - m.m[0][0] - m.m[1][1]);
                q.w = (m.m[1][0] - m.m[0][1]) / s;
                q.x = (m.m[0][2] + m.m[2][0]) / s;
                q.y = (m.m[1][2] + m.m[2][1]) / s;
                q.z = 0.25f * s;
            }
        }

        q.normalize();
        return q;
    }

    void calculateOrientation() {
        // Build inertia tensor matrix
        Matrix3x3 inertiaTensor;

        // Initialize to zero
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                inertiaTensor.m[i][j] = 0.0f;
            }
        }

        // Calculate centered vertices
        std::vector<float3> centeredVertices;
        for (const auto &v: vertices) {
            centeredVertices.push_back(v - centerOfMass);
        }

        // Compute inertia tensor
        float totalMass = 1.0f; // Uniform density
        float particleMass = totalMass / centeredVertices.size();

        for (const auto &v: centeredVertices) {
            float x = v.x, y = v.y, z = v.z;
            inertiaTensor.m[0][0] += particleMass * (y * y + z * z);
            inertiaTensor.m[1][1] += particleMass * (x * x + z * z);
            inertiaTensor.m[2][2] += particleMass * (x * x + y * y);
            inertiaTensor.m[0][1] -= particleMass * x * y;
            inertiaTensor.m[0][2] -= particleMass * x * z;
            inertiaTensor.m[1][2] -= particleMass * y * z;
        }

        // Mirror symmetric elements
        inertiaTensor.m[1][0] = inertiaTensor.m[0][1];
        inertiaTensor.m[2][0] = inertiaTensor.m[0][2];
        inertiaTensor.m[2][1] = inertiaTensor.m[1][2];

        // Calculate eigenvalues and eigenvectors
        float eigenvalues[3];
        Matrix3x3 eigenvectors;
        computeEigenvaluesAndEigenvectors(inertiaTensor, eigenvalues, eigenvectors);

        // Convert to quaternion
        orientation = matrixToQuaternion(eigenvectors);
    }


    std::vector<float3> vertices;

    std::vector<float3> normals;

    std::vector<Face> faces;

    float3 centerOfMass = {0.0f, 0.0f, 0.0f};

    float volume = 0.0f;

    float3 minPt;

    float3 maxPt;

    Quaternion orientation;
};


#endif //POLYTOPE_H
