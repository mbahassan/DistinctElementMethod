//
// Created by iqraa on 28-2-25.
//

#ifndef QUADTREEBUILDER_H
#define QUADTREEBUILDER_H

#include <Particle/Spherical.hpp>

#include "QuadTree.h"
#include "ContactDetection/BroadPhase/ITreeBuilder.h"
#include <ContactDetection/BroadPhase/Config/TreeConfig.h>


class QuadTreeBuilder : public ITreeBuilder<QuadTree, Spherical>
{
    public:
    QuadTreeBuilder(const TreeConfig& treeConfig);

    ~QuadTreeBuilder() override;

    void initialize(const int capacity) override;

    void build(Spherical* point, const int size) override;

    void reset() override;

    private:
    Spherical* pointsExch;

    TreeConfig treeConfig;
};



#endif //QUADTREEBUILDER_H
