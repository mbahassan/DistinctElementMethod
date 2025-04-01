//
// Created by iqraa on 26-2-25.
//

#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include <regex>
#include <fstream>
#include <iostream>
#include "Tools/Config/Config.h"
#include "Tools/nlohmann/json.hpp"

class Parser
{
  public:
    static Config getConfig(const std::string& path)
    {
        const nlohmann::json data = readJson(path);

        if (data.empty()) { return {}; }

        Config result;
        result.restart = true;
        result.numberOfParticles = data["Simulation"]["numberOfParticles"];

        result.materialPath= data["Simulation"]["materialPath"];
        result.shapePath = data["Simulation"]["shapePath"];

        result.treeType_ = data["ContactDetection"]["BroadPhase"]["treeType"];
        result.maxDepth = data["ContactDetection"]["BroadPhase"]["maxDepth"];
        result.minPointsPerNode = data["ContactDetection"]["BroadPhase"]["minPointsPerNode"];


        solveRelativePath(result.materialPath, path);
        solveRelativePath(result.shapePath, path);

        return result;
    }

    static nlohmann::json readJson(const std::string& path)
    {
        nlohmann::json config;
        std::ifstream inpStream(path);
        if (!inpStream.is_open())
        {
            std::cerr << "file not found: " << path << std::endl;
            return {};
        }

        inpStream >> config;
        inpStream.close();

        return config;
    }

    private:

    static void solveRelativePath(std::string& path, const std::string& parentPath)
    {
        std::regex configNameRegex("\\./.+\\.json");
        std::smatch relativePathMatch;

        if (path[0] == '.')
        {
            std::regex parentFolderRegex("[a-zA-Z0-9]+.json");
            std::smatch parentFolderMatch;
            if (std::regex_search(parentPath, parentFolderMatch, parentFolderRegex) &&
                parentFolderMatch.prefix().length() > 0)
            {
                path = parentFolderMatch.prefix().str() + path.substr(2);
            }
        }
    }


};







#endif //CONFIGPARSER_H
