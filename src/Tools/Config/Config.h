//
// Created by iqraa on 25-2-25.
//

#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <ContactDetection/BroadPhase/Config/TreeType.h>


struct Config
{
  /// Total number of particles
  int numberOfParticles;

  /// Collection of the needed paths
  std::string shapePath;
  std::string materialPath;


  /// Contact Detection
  TreeType treeType_;
  int maxDepth;
  int minPointsPerNode;


  /// Restarting simulation default yes
  bool restart = true;
};

#endif //CONFIG_H
