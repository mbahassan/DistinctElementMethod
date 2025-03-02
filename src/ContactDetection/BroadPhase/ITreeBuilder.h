//
// Created by iqraa on 28-2-25.
//

#ifndef ITREEBUILDER_H
#define ITREEBUILDER_H

template<typename T, typename P>
class ITreeBuilder
{

public:
    virtual ~ITreeBuilder() = default;

    virtual void initialize(const int capacity) = 0;

    virtual void build(P* points, const int count) = 0;

    virtual void reset() = 0;

    const T& getTree() const
    {
        return *tree;
    }

protected:
    T* tree;
};

#endif //ITREEBUILDER_H
