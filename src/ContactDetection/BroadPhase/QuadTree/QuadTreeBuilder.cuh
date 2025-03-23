//
// Created by iqraa on 28-2-25.
//

#ifndef QUADTREEBUILDER_H
#define QUADTREEBUILDER_H

#include <Particle/Spherical.hpp>
#include <Particle/Polyhedral.hpp>

#include "QuadTree.h"
#include "ContactDetection/BroadPhase/ITreeBuilder.h"
#include <ContactDetection/BroadPhase/Config/TreeConfig.h>

template<typename ParticleType>
class QuadTreeBuilder : public ITreeBuilder<QuadTree, ParticleType>
{
    public:
    QuadTreeBuilder(const TreeConfig& treeConfig);

    ~QuadTreeBuilder() override;

    void initialize(int capacity) override;

    void build(ParticleType* point, int size) override;

    void reset() override;

    private:
    ParticleType* pointsExch;

    TreeConfig treeConfig;
};

template class QuadTreeBuilder<Spherical>;
// template class QuadTreeBuilder<Polyhedral>;

#endif //QUADTREEBUILDER_H
