//
// Created by iqraa on 25-2-25.
//

#ifndef CONFIGPARSERBASE_H
#define CONFIGPARSERBASE_H

#include <string>
#include <fstream>
#include <iostream>
#include <Tools/nlohmann/json.h>

nlohmann::json ParseBase(const std::string& path)
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

#endif //CONFIGPARSERBASE_H
