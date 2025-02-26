//
// Created by iqraa on 26-2-25.
//

#ifndef CONFIGPARSER_H
#define CONFIGPARSER_H

#include "ConfigParser.h"
#include <Toools/nlohmann/json.h>

void SolveRelativePath(std::string& path, const std::string& parentPath)
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

    // if (std::regex_search(path, relativePathMatch, configNameRegex) &&
    //     relativePathMatch.prefix().length() == 0)
    // {
    //     std::regex parentFolderRegex("[a-zA-Z0-9]+.json");
    //     std::smatch parentFolderMatch;
    //     if (std::regex_search(parentPath, parentFolderMatch, parentFolderRegex) &&
    //         parentFolderMatch.prefix().length() > 0)
    //     {
    //         path = parentFolderMatch.prefix().str() + path.substr(2);
    //     }
    // }
}

AppConfig ParseAppConfig(const std::string& path)
{
    const nlohmann::json data = ParseBase(path);

    if (data.empty()) { return {}; }

    AppConfig result;
    result.isValid = true;
    result.pointsCount = data["points_count"];
    result.treeConfigPath = data["tree_config"];
    result.renderConfigPath = data["render_config"];
    result.modelPath = data["model_path"];
    result.enableRender = data["enable_render"];

    SolveRelativePath(result.treeConfigPath, path);
    SolveRelativePath(result.renderConfigPath, path);
    SolveRelativePath(result.modelPath, path);

    return result;
}

#endif //CONFIGPARSER_H
