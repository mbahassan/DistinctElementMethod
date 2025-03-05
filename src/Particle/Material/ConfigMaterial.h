//
// Created by mbahassan on 2/26/25.
//

#ifndef CONFIGMATERIAL_H
#define CONFIGMATERIAL_H

#include <string>

struct ConfigMaterial
{
    /// Collection of the needed paths
    int materialId;

    float youngsModulus;
    float density;
    float poissonRatio;
    float frictionCoeff;
    float restitutionCoeff;
};

#endif //CONFIGMATERIAL_H
