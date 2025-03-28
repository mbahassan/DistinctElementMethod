#ifndef OBJREADER_H
#define OBJREADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <vector_types.h>
#include <Tools/quaternion/quaternion.hpp>
// #include <Tools/Vector.h>

struct Face {
    std::vector<int> Indices; // 3 or 4
};

class ObjReader {
public:
    /// Default Constructor
    ObjReader() = default;

    /// Constructor from File
    explicit ObjReader(const std::string& filename)
    {
        load(filename);
        volume = computeVolume();
        com = computeCOM();
        orientation = computeOrientation();
        computeBoundingBox();
    }

    /// Copy constructor and assignment operator can use default
    ObjReader(const ObjReader&) = default;
    ObjReader& operator=(const ObjReader&) = default;

    /// Destructor
    ~ObjReader() = default;

bool load(const std::string& filename) {
    vertices.clear();
    faces.clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open OBJ file.\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            float x, y, z;
            if (iss >> x >> y >> z) {
                vertices.push_back({x, y, z});
            }
        } else if (token == "f") {
            Face face;
            std::string vertexInfo;

            while (iss >> vertexInfo) {
                // Split the vertex info (v/vt/vn)
                std::istringstream vertexStream(vertexInfo);
                std::string indexStr;

                // Extract first part (vertex index)
                std::getline(vertexStream, indexStr, '/');

                try {
                    int vertexIndex = std::stoi(indexStr);

                    // Adjust for 1-based indexing in OBJ files
                    vertexIndex -= 1;

                    // Validate index
                    if (vertexIndex >= 0 && vertexIndex < static_cast<int>(vertices.size())) {
                        face.Indices.push_back(vertexIndex);
                    } else {
                        std::cerr << "Invalid vertex index: " << vertexIndex + 1
                                  << " (max vertices: " << vertices.size() << ")\n";
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing face index: " << vertexInfo << " - " << e.what() << "\n";
                }
            }

            // Only add faces with 3 or 4 vertices (triangles or quads)
            if (face.Indices.size() == 3 || face.Indices.size() == 4) {
                faces.push_back(face);
            } else {
                std::cerr << "Skipping face with " << face.Indices.size() << " vertices\n";
            }
        }
    }

    file.close();

    // Debug output
    std::cout << "Loaded " << vertices.size() << " vertices" << std::endl;
    std::cout << "Loaded " << faces.size() << " faces" << std::endl;

    for (size_t i = 0; i < faces.size(); ++i) {
        std::cout << "Face " << i << " indices: ";
        for (int idx : faces[i].Indices) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
    }

    return true;
}

    const std::vector<Face>& getFaces() const { return faces; }


    const std::vector<float3>& getVertices() const {return vertices;}


    void printSummary() const
    {
        std::cout << "Vertices: " << vertices.size() << "\n";
        std::cout << "Faces: " << faces.size() << "\n";
        for (const auto& v : vertices)
            std::cout << "v " << v.x << " " << v.y << " " << v.z << "\n";
        for (const auto& face : faces) {
            std::cout << "f ";
            for (int j = 0; j < face.Indices.size(); ++j)
                std::cout << face.Indices[j] + 1 << " ";
            std::cout << "\n";
        }
    }

    float getObjVolume() const { return volume; }
    float3 getObjMin() const { return outMin; }
    float3 getObjMax() const { return outMax; }
    Quaternion getObjOrientation() const { return orientation; }
    float3 getObjCOM() const { return com; }

private:
    // Modify volume and COM calculations to handle triangles/quads dynamically
    float computeVolume() const
    {
        float vol = 0.0f;

        for (auto& face : faces) {
            // Only process triangles and quads
            if (face.Indices.size() == 3 || face.Indices.size() == 4) {
                auto idx = face.Indices[0];
                float3 v0 = vertices[idx];

                // Triangulate for quads
                int endIndex = (face.Indices.size() == 4) ? 3 : 2;
                for (int j = 1; j < endIndex; ++j) {
                    float3 v1 = vertices[face.Indices[j]];
                    float3 v2 = vertices[face.Indices[j + 1]];

                    // Signed volume of tetrahedron
                    float detMatrix = v0.x * (v1.y * v2.z - v2.y * v1.z) +
                                      v0.y * (v2.x * v1.z - v1.x * v2.z) +
                                      v0.z * (v1.x * v2.y - v2.x * v1.y);

                    vol += detMatrix;
                }
            }
        }

        // Volume is 1/6 of the sum
        return std::abs(vol) / 6.0f;
    }


    void computeBoundingBox() {
        if (vertices.empty()) {
            std::cerr << "Cannot get BBox!" << std::endl;
            return;
        }
        // Initialize to the first vertex
        outMin = vertices[0];
        outMax = vertices[0];
        // Iterate over all vertices to find the min and max values

        for (int i = 0; i < vertices.size(); i++) {
            auto v = vertices[i];
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
        float3 center = computeCOM();

        // Compute covariance matrix
        float covMat[3][3] = {{0.0f}};

        for (const auto& vertex : vertices) {
            float dx = vertex.x - center.x;
            float dy = vertex.y - center.y;
            float dz = vertex.z - center.z;

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
        size_t vertexCount = vertices.size();
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
        float3 center = {0.0f, 0.0f, 0.0f};
        float totalVolume = 0.0f;

        // For each face
        for (const auto& face : faces) {
            // Only process triangular faces
            if (face.Indices.size() >= 3) {
                float3 v0 = vertices[face.Indices[0]];

                // For triangles and quads, we can triangulate
                for (int j = 1; j < face.Indices.size() - 1; ++j) {
                    float3 v1 = vertices[face.Indices[j]];
                    float3 v2 = vertices[face.Indices[j + 1]];

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
                    center.x += centroid.x * volume;
                    center.y += centroid.y * volume;
                    center.z += centroid.z * volume;
                }
            }
        }

        // Normalize by total volume
        if (totalVolume != 0.0f) {
            center.x /= totalVolume;
            center.y /= totalVolume;
            center.z /= totalVolume;
        }

        return center;
    }

    // Member variables
    std::vector<float3> vertices;
    std::vector<Face> faces;
    Quaternion orientation;

    float3 outMin {0.0f, 0.0f, 0.0f};
    float3 outMax {0.0f, 0.0f, 0.0f};

    float3 com {0.0f, 0.0f, 0.0f};
    float volume = 0.0f;
};

#endif //OBJREADER_H