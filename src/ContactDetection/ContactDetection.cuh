//
// Created by mbahassan on 2/28/25.
//

#ifndef CONTACTDETECTION_CUH
#define CONTACTDETECTION_CUH

#include <unordered_set>
#include <Output/QuadTreeWriter.cuh>

#include "BroadPhase/QuadTree/QuadTreeBuilder.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "Particle/Particle.hpp"
#include "Tools/CudaHelper.hpp"

struct PotentialContact
{
    int nodeId;
    int particleIdI;
    int particleIdJ;
};

struct Contact
{
    Particle pi;
    Particle pj;
    float3 normal;
    float3 contactPoint;
};

class ContactDetection
{
    public:
    explicit ContactDetection(const TreeType treeType): treeConfig_(2,1)
    {
        treeType_ = treeType;
    }

    std::vector<PotentialContact> broadPhase(std::vector<Particle>& particles)
    {
        detectContacts(particles);

        return findPotentialContacts(particles);
    }

    std::vector<Contact> narrowPhase(std::vector<PotentialContact>& potentialContact)
    {

        return {};
    }

    void detectContacts(std::vector<Particle>& particles)
    {
        if (treeType_ == QUADTREE)
        {
            int particlesCount = particles.size();
            Particle* pointsHost = particles.data();
            std::cout << "RunAppOctree(): " << particlesCount << "\n";
            Particle* points;
            hostToDevice(pointsHost, particlesCount, &points);

            treeBuilder = std::make_unique<QuadTreeBuilder>(treeConfig_);
            treeBuilder->initialize(particlesCount);
            treeBuilder->build(points, particlesCount);

            QuadTreeWriter::writeQuadTree("./results/quadtree_000000.vtu", &treeBuilder->getTree(), treeConfig_);
            deviceToHost(points, particlesCount, &pointsHost);

        }
    }


private:

    std::vector<PotentialContact> findPotentialContacts(std::vector<Particle>& points);

    void checkContactsInLeaf(const QuadTree* leaf, std::vector<Particle>& points,
        std::vector<PotentialContact>& contacts);

    void checkContactsBetweenLeaves(
    const QuadTree* leafA,
    const QuadTree* leafB,
    std::vector<Particle>& points,
    std::vector<PotentialContact>& contacts,
    std::unordered_set<uint64_t>& processedPairs);

    bool areNeighboringNodes(const QuadTree* nodeI, const QuadTree* nodeJ);

    TreeType treeType_;

    TreeConfig treeConfig_ ;

    std::unique_ptr<QuadTreeBuilder> treeBuilder;
};



#endif //CONTACTDETECTION_CUH
