//
// Created by iqraa on 23-3-25.
//

#ifndef OBJREADER_H
#define OBJREADER_H

#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector_types.h>
#include <Tools/quaternion/quaternion.hpp>

struct Face {
    int indices[4];  // support up to quad
    int size;        // 3 or 4
};

class ObjReader {
public:
    /// Default Constructor
    // ObjReader() = default;


    /// Constructor form File
    explicit ObjReader(const std::string& filename)
    {
        load(filename);
        volume = computeVolume();
        com = computeCOM();
        orientation = computeOrientation();
        computeBoundingBox();
    }

    /// Destructor
    virtual ~ObjReader()
    {
        delete[] vertices;
        delete[] faces;
    }

    bool load(const std::string& filename) {
        int estimatedVertices = 0;
        int estimatedFaces = 0;

        // First pass to count vertices and faces
        {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open OBJ file.\n";
                return false;
            }

            std::string line;
            while (std::getline(file, line)) {
                if (line.rfind("v ", 0) == 0)
                    estimatedVertices++;
                else if (line.rfind("f ", 0) == 0)
                    estimatedFaces++;
            }
            file.close();
        }

        vertexCount = estimatedVertices;
        faceCount = estimatedFaces;

        // Allocate arrays dynamically
        vertices = new float3[vertexCount];
        faces = new Face[faceCount];

        // Second pass: actually read data
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open OBJ file on second pass.\n";
            return false;
        }

        std::string line;
        int vIndex = 0, fIndex = 0;
        while (std::getline(file, line)) {
            std::istringstream iss(line);

            if (line.rfind("v ", 0) == 0) {
                char v;
                float x, y, z;
                iss >> v >> x >> y >> z;
                vertices[vIndex++] = {x, y, z};

            } else if (line.rfind("f ", 0) == 0) {
                char f;
                std::string token;
                Face face;
                face.size = 0;

                iss >> f;
                while (iss >> token && face.size < 4) {
                    std::istringstream tokenStream(token);
                    std::string idxStr;
                    std::getline(tokenStream, idxStr, '/');
                    int idx = std::stoi(idxStr) - 1;
                    face.indices[face.size++] = idx;
                }

                faces[fIndex++] = face;
            }
        }

        file.close();
        return true;
    }

    int getVerticesCount() const { return vertexCount; }

    int getFacesCount() const {return faceCount;}

    float3 getVertex(int i) const {return vertices[i];}

    Face& getFace(int i) const {
        return faces[i];
    }

    void printSummary() const
    {
        std::cout << "Vertices: " << vertexCount << "\n";
        std::cout << "Faces: " << faceCount << "\n";
        for (int i = 0; i < vertexCount; ++i)
            std::cout << "v " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << "\n";
        for (int i = 0; i < faceCount; ++i) {
            std::cout << "f ";
            for (int j = 0; j < faces[i].size; ++j)
                std::cout << faces[i].indices[j] + 1 << " "; // Fixed: was faces[i].indices[i]
            std::cout << "\n";
        }
    }

    float3* getVertices() const { return vertices; }

    Face* getFaces() const { return faces; }

    // Calculate volume of the mesh using the divergence theorem
    float computeVolume() const {
        float volume = 0.0f;

        // For each face
        for (int i = 0; i < faceCount; ++i) {
            const Face& face = faces[i];

            // Only process triangular faces
            if (face.size >= 3) {
                float3 v0 = vertices[face.indices[0]];

                // For triangles and quads, we can triangulate
                for (int j = 1; j < face.size - 1; ++j) {
                    float3 v1 = vertices[face.indices[j]];
                    float3 v2 = vertices[face.indices[j + 1]];

                    // Signed volume of tetrahedron
                    float detMatrix = v0.x * (v1.y * v2.z - v2.y * v1.z) +
                                      v0.y * (v2.x * v1.z - v1.x * v2.z) +
                                      v0.z * (v1.x * v2.y - v2.x * v1.y);

                    volume += detMatrix;
                }
            }
        }

        // Volume is 1/6 of the sum
        return std::abs(volume) / 6.0f;
    }

    float getObjVolume() const {return volume;}

    float3 getObjMin() const{return outMin;}

    float3 getObjMax() const{return outMax;}

    Quaternion getObjOrientation() const {return orientation;}

    float3 getObjCOM() const {return com;}

private:
    // New function to compute the axis-aligned bounding box:
    // It returns the minimum and maximum points of the bounding box through output parameters.
    void computeBoundingBox() {
        if (vertexCount == 0) {
            std::cerr << "Can not get BBox !" << std::endl;
            return;
        }
        // Initialize to the first vertex
        outMin = vertices[0];
        outMax = vertices[0];
        // Iterate over all vertices to find the min and max values
        for (int i = 1; i < vertexCount; ++i) {
            float3 v = vertices[i];
            if (v.x < outMin.x) outMin.x = v.x;
            if (v.y < outMin.y) outMin.y = v.y;
            if (v.z < outMin.z) outMin.z = v.z;
            if (v.x > outMax.x) outMax.x = v.x;
            if (v.y > outMax.y) outMax.y = v.y;
            if (v.z > outMax.z) outMax.z = v.z;
        }
    }

    // Calculate orientation as quaternion using Principal Component Analysis (PCA)
    Quaternion computeOrientation() const {
        // Calculate center of mass first
        float3 com = computeCOM();

        // Compute covariance matrix
        float covMat[3][3] = {{0.0f}};

        for (int i = 0; i < vertexCount; ++i) {
            float dx = vertices[i].x - com.x;
            float dy = vertices[i].y - com.y;
            float dz = vertices[i].z - com.z;

            covMat[0][0] += dx * dx;
            covMat[0][1] += dx * dy;
            covMat[0][2] += dx * dz;
            covMat[1][0] += dy * dx;
            covMat[1][1] += dy * dy;
            covMat[1][2] += dy * dz;
            covMat[2][0] += dz * dx;
            covMat[2][1] += dz * dy;
            covMat[2][2] += dz * dz;
        }

        // Normalize
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covMat[i][j] /= vertexCount;
            }
        }

        // Simple approach to find eigenvectors (in practice, you'd use a library)
        // For simplicity, we use the power iteration method to find the dominant eigenvector
        float vec[3] = {1.0f, 0.0f, 0.0f}; // Initial guess

        // Iterate to converge on principal eigenvector
        for (int iter = 0; iter < 20; ++iter) {
            float newVec[3] = {0.0f};

            // Multiply by covariance matrix
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    newVec[i] += covMat[i][j] * vec[j];
                }
            }

            // Normalize
            float length = std::sqrt(newVec[0]*newVec[0] + newVec[1]*newVec[1] + newVec[2]*newVec[2]);
            for (int i = 0; i < 3; ++i) {
                vec[i] = newVec[i] / length;
            }
        }

        // Converting principal axis to quaternion
        // We assume [0,0,1] is the "up" direction for the object's natural orientation
        float3 zAxis = {0.0f, 0.0f, 1.0f};
        float3 axis;

        // Cross product to find rotation axis
        axis.x = zAxis.y * vec[2] - zAxis.z * vec[1];
        axis.y = zAxis.z * vec[0] - zAxis.x * vec[2];
        axis.z = zAxis.x * vec[1] - zAxis.y * vec[0];

        // Normalize rotation axis
        float axisLength = std::sqrt(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);

        // Handle parallel case
        if (axisLength < 1e-6f) {
            // Check if vectors are parallel or anti-parallel
            float dot = zAxis.x * vec[0] + zAxis.y * vec[1] + zAxis.z * vec[2];
            if (dot > 0.0f) {
                // Parallel, no rotation needed
                return {1.0f, 0.0f, 0.0f, 0.0f};
            } else {
                // Anti-parallel, 180° rotation around x-axis
                return {0.0f, 1.0f, 0.0f, 0.0f};
            }
        }

        // Normalize axis
        axis.x /= axisLength;
        axis.y /= axisLength;
        axis.z /= axisLength;

        // Calculate rotation angle
        float dotProduct = zAxis.x * vec[0] + zAxis.y * vec[1] + zAxis.z * vec[2];
        float angle = std::acos(dotProduct);

        // Create quaternion
        Quaternion q;
        q.w = std::cos(angle / 2.0f);
        float sinHalfAngle = std::sin(angle / 2.0f);
        q.x = axis.x * sinHalfAngle;
        q.y = axis.y * sinHalfAngle;
        q.z = axis.z * sinHalfAngle;

        return q;
    }

    // Calculate center of mass (centroid)
    float3 computeCOM() const {
        float3 com = {0.0f, 0.0f, 0.0f};
        float totalVolume = 0.0f;

        // For each face
        for (int i = 0; i < faceCount; ++i) {
            const Face& face = faces[i];

            // Only process triangular faces
            if (face.size >= 3) {
                float3 v0 = vertices[face.indices[0]];

                // For triangles and quads, we can triangulate
                for (int j = 1; j < face.size - 1; ++j) {
                    float3 v1 = vertices[face.indices[j]];
                    float3 v2 = vertices[face.indices[j + 1]];

                    // Signed volume of tetrahedron
                    float detMatrix = v0.x * (v1.y * v2.z - v2.y * v1.z) +
                                      v0.y * (v2.x * v1.z - v1.x * v2.z) +
                                      v0.z * (v1.x * v2.y - v2.x * v1.y);

                    // Volume contribution
                    float volume = detMatrix / 6.0f;
                    totalVolume += volume;

                    // Center of current tetrahedron (centroid is at 1/4 of the way from origin to each vertex)
                    float3 centroid;
                    centroid.x = (v0.x + v1.x + v2.x) / 4.0f;
                    centroid.y = (v0.y + v1.y + v2.y) / 4.0f;
                    centroid.z = (v0.z + v1.z + v2.z) / 4.0f;

                    // Weighted contribution to COM
                    com.x += centroid.x * volume;
                    com.y += centroid.y * volume;
                    com.z += centroid.z * volume;
                }
            }
        }

        // Normalize by total volume
        if (totalVolume != 0.0f) {
            com.x /= totalVolume;
            com.y /= totalVolume;
            com.z /= totalVolume;
        }

        return com;
    }


    float3* vertices = nullptr;
    Face* faces = nullptr;
    Quaternion orientation;

    int vertexCount = 0;
    int faceCount = 0;

    float3 outMin = {0.0f, 0.0f, 0.0f};
    float3 outMax = {0.0f, 0.0f, 0.0f};

    float3 com;
    float volume;
};


#endif //OBJREADER_H
