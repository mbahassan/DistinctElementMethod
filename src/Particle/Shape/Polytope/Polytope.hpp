#ifndef POLYTOPE_H
#define POLYTOPE_H
#include <cfloat>
#include <string>
#include <vector>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>

#include "Tools/objreader/ObjReader.h"
#include "Tools/quaternion/quaternion.hpp"

class Polytope : public Shape {
public:
    Polytope()
    {
        setShapeType(POLYHEDRAL);
    }

    /// Constructor from File
    explicit Polytope(const std::string &file)
    {
        setShapeType(POLYHEDRAL);
        objReader = ObjReader(file);

        // Store the computed values from ObjReader
        polytopeVolume = objReader.getObjVolume();
        polytopeCOM = objReader.getObjCOM();
        polytopeOrientation = objReader.getObjOrientation();
        polytopeMin = objReader.getObjMin();
        polytopeMax = objReader.getObjMax();
        
        // Deep copy the vertices and faces
        copyVerticesAndFaces();
    }

    // Copy constructor
    Polytope(const Polytope& other) : Shape(other)
    {
        copyFrom(other);
    }
    
    // Assignment operator
    __host__ __device__
    Polytope& operator=(const Polytope& other)
    {
        if (this != &other) {
            Shape::operator=(other);
            cleanup();
            copyFrom(other);
        }
        return *this;
    }

    ~Polytope() override
    {
        cleanup();
    }

    float3 supportMapping(const float3 &direction) const
    {
        if (vertexCount == 0) return {0.0f, 0.0f, 0.0f};
        
        float maxDistance = vertices[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < vertexCount; i++) {
            float distance = vertices[i] & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }

        return vertices[maxIdx];
    }

    __host__ __device__
    float getVolume() override { return polytopeVolume; }

    __host__ __device__
    float3 getMin() override { return polytopeMin; }

    __host__ __device__
    float3 getMax() override { return polytopeMax; }

    __host__ __device__
    float3 getCenterOfMass() const { return polytopeCOM; }

    // Method to get the orientation
    __host__ __device__
    Quaternion getOrientation() const {
        return polytopeOrientation;
    }

    // Get number of vertices
    int getVerticesCount() const { return vertexCount; }

    // Get vertex at index
    float3 getVertex(int i) const {
        if (i >= 0 && i < vertexCount) {
            return vertices[i];
        }
        // Return default if out of bounds
        return {0.0f, 0.0f, 0.0f};
    }

    // Get number of faces
    int getFacesCount() const { return faceCount; }

    // Get face at index
    Face getFace(int i) const {
        if (i >= 0 && i < faceCount) {
            return faces[i];
        }
        // Return empty face if out of bounds
        Face emptyFace;
        emptyFace.size = 0;
        for (int j = 0; j < 4; j++) {
            emptyFace.indices[j] = 0;
        }
        return emptyFace;
    }

    // Debug method to validate face data
    void validateFaces() const {
        for (int i = 0; i < faceCount; i++) {
            if (faces[i].size < 0 || faces[i].size > 4) {
                std::cerr << "Invalid face size at index " << i << ": " << faces[i].size << std::endl;
            } else {
                for (int j = 0; j < faces[i].size; j++) {
                    if (faces[i].indices[j] < 0 || faces[i].indices[j] >= vertexCount) {
                        std::cerr << "Invalid vertex index at face " << i << ", vertex " << j 
                                  << ": " << faces[i].indices[j] << " (max: " << vertexCount-1 << ")" << std::endl;
                    }
                }
            }
        }
    }

private:
    __host__ __device__
    void cleanup() {
        delete[] vertices;
        delete[] faces;
        vertices = nullptr;
        faces = nullptr;
        vertexCount = 0;
        faceCount = 0;
    }

    __host__ __device__
    void copyFrom(const Polytope& other) {
        // objReader = other.objReader;
        polytopeVolume = other.polytopeVolume;
        polytopeCOM = other.polytopeCOM;
        polytopeOrientation = other.polytopeOrientation;
        polytopeMin = other.polytopeMin;
        polytopeMax = other.polytopeMax;
        
        vertexCount = other.vertexCount;
        faceCount = other.faceCount;
        
        if (vertexCount > 0) {
            vertices = new float3[vertexCount];
            for (int i = 0; i < vertexCount; i++) {
                vertices[i] = other.vertices[i];
            }
        }
        
        if (faceCount > 0) {
            faces = new Face[faceCount];
            for (int i = 0; i < faceCount; i++) {
                faces[i] = other.faces[i];
            }
        }
    }
    
    void copyVerticesAndFaces() {
        // Release any existing data
        cleanup();
        
        // Get counts
        vertexCount = objReader.getVerticesCount();
        faceCount = objReader.getFacesCount();
        
        // Allocate new memory
        if (vertexCount > 0) {
            vertices = new float3[vertexCount];
            float3* srcVertices = objReader.getVertices();
            for (int i = 0; i < vertexCount; i++) {
                vertices[i] = srcVertices[i];
            }
        }
        
        if (faceCount > 0) {
            faces = new Face[faceCount];
            Face* srcFaces = objReader.getFaces();
            for (int i = 0; i < faceCount; i++) {
                faces[i] = srcFaces[i];
            }
        }
    }

    ObjReader objReader;
    float polytopeVolume = 0.0f;
    float3 polytopeCOM {0.0f, 0.0f, 0.0f};
    Quaternion polytopeOrientation;
    float3 polytopeMin {0.0f, 0.0f, 0.0f};
    float3 polytopeMax {0.0f, 0.0f, 0.0f};
    
    // Owned data (deep copy from ObjReader)
    float3* vertices = nullptr;
    Face* faces = nullptr;
    int vertexCount = 0;
    int faceCount = 0;
};

#endif //POLYTOPE_H