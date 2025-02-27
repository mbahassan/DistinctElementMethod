//
// Created by iqraa on 25-2-25.
//

#ifndef CONFIG_H
#define CONFIG_H

#include <string>


struct Config
{
  /// Total number of particles
  int numberOfParticles;

  /// Collection of the needed paths
  std::string shapeConfigPath;
  std::string materialConfigPath;


  /// Restarting simulation default yes
  bool restart = true;
};

#endif //CONFIG_H
