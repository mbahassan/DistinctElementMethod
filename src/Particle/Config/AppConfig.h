//
// Created by iqraa on 25-2-25.
//

#ifndef APPCONFIG_H
#define APPCONFIG_H

#include "Particle/Config/BaseConfig.h"
#include <string>
struct AppConfig : public BaseConfig
{
    std::string particleConfigPath;
    std::string modelPath;
};

#endif //APPCONFIG_H
