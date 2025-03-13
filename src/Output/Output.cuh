//
// Created by iqraa on 5-3-25.
//

#ifndef OUTPUT_H
#define OUTPUT_H

#include <vector>
#include <ContactDetection/BroadPhase/QuadTree/QuadTree.h>
#include <Particle/Particle.hpp>

class Output {
public:
    Output(const std::string& dir):
    output_dir(dir) {
        // Ensure output directory exists
        std::filesystem::create_directories(dir);
    }

    ~Output() = default;

    void writeParticles(const std::vector<Particle>& particles, int timestep);

    void writeTree(const QuadTree* quadtree, int timestep);

private:
    std::string output_dir;

    // Create filename with zero-padded timestep
    std::string createFilename(
        const std::string& prefix,
        int timestep,
        const std::string& extension
    ) {
        std::ostringstream filename;
        filename << output_dir << "/"
                 << prefix << "_"
                 << std::setfill('0') << std::setw(6) << timestep
                 << extension;
        return filename.str();
    }
};


#endif //OUTPUT_H
