//
// Created by iqraa on 12-3-25.
//

#ifndef EPA_CUH
#define EPA_CUH



class EPA {
public:
    std::pair<float3, float> computePenetration(const Particle& A, const Particle& B, const Simplex& gjkSimplex);
private:
    struct Triangle
    {
        int indices[3];  // Indices to vertices in the polytope
        float3 normal;   // Outward facing normal
        float distance;  // Distance from origin to face along normal

        Triangle(int a, int b, int c, const std::vector<float3>& vertices);

        bool isFrontFacing(const float3& point, const std::vector<float3>& vertices) const;
    };

    struct Edge
    {
        int a, b;

        Edge(int a_, int b_);

        bool operator==(const Edge& other) const;

        bool operator<(const Edge& other) const;
    };

    float3 supportFunction(const Particle& A, const Particle& B, const float3& d);
};



#endif //EPA_CUH
