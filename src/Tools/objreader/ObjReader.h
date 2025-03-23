//
// Created by iqraa on 23-3-25.
//

#ifndef OBJREADER_H
#define OBJREADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector_types.h>

class ObjReader {
public:
    float3* vertices = nullptr;
    int (*faces)[4] = nullptr;
    int* faceVertexCount = nullptr;

    int vertexCount = 0;
    int faceCount = 0;

    ~ObjReader() {
        delete[] vertices;
        delete[] faces;
        delete[] faceVertexCount;
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
        faces = new int[faceCount][4];
        faceVertexCount = new int[faceCount];

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
                std::string tokens[4];
                iss >> f >> tokens[0] >> tokens[1] >> tokens[2] >> tokens[3];

                int indices[4] = {-1, -1, -1, -1};
                int nVerts = (tokens[3].empty() ? 3 : 4);

                for (int i = 0; i < nVerts; ++i) {
                    std::istringstream ts(tokens[i]);
                    std::string idxStr;
                    std::getline(ts, idxStr, '/');
                    indices[i] = std::stoi(idxStr) - 1;
                }

                for (int i = 0; i < nVerts; ++i)
                    faces[fIndex][i] = indices[i];

                faceVertexCount[fIndex] = nVerts;
                fIndex++;
            }
        }

        file.close();
        return true;
    }

    void printSummary() const {
        std::cout << "Vertices: " << vertexCount << "\n";
        std::cout << "Faces: " << faceCount << "\n";
        for (int i = 0; i < vertexCount; ++i)
            std::cout << "v " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << "\n";
        for (int i = 0; i < faceCount; ++i) {
            std::cout << "f ";
            for (int j = 0; j < faceVertexCount[i]; ++j)
                std::cout << faces[i][j] + 1 << " ";
            std::cout << "\n";
        }
    }

};



#endif //OBJREADER_H
