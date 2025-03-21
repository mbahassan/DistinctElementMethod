//
// Created by iqraa on 28-2-25.
//

#ifndef OCTREEBUILDER_H
#define OCTREEBUILDER_H

#include <Particle/Spherical.hpp>

#include "Octree.h"
#include <ContactDetection/BroadPhase/ITreeBuilder.h>
#include <ContactDetection/BroadPhase/Config/TreeConfig.h>


class OctreeBuilder : public ITreeBuilder<Octree, Spherical>
{
    public:
    OctreeBuilder(const TreeConfig& treeConfig);

    ~OctreeBuilder() override;

    void initialize(const int capacity) override;

    void build(Spherical* point, const int size) override;

    void reset() override;

    private:
    Spherical* pointsExch;

    TreeConfig treeConfig;
};



#endif //OCTREEBUILDER_H
