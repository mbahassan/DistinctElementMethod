//
// Created by iqraa on 5-3-25.
//

#include "Output.cuh"

#include "QuadTreeWriter.cuh"

void Output::writeParticles(const std::vector<Particle> &particles, const int timestep) {

    // Create filename with zero-padded timestep
    std::string filename = createFilename("particles", timestep, ".vtp");
    std::ofstream vtpFile(filename);

    if (!vtpFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

   // XML Header
        vtpFile << "<?xml version=\"1.0\"?>\n";
        vtpFile << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        vtpFile << "  <PolyData>\n";
        vtpFile << "    <Piece NumberOfPoints=\"" << particles.size() << "\" NumberOfVerts=\"" << particles.size() << "\">\n";

        // Points section
        vtpFile << "      <Points>\n";
        vtpFile << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const auto& particle : particles) {
            vtpFile << "          "
                    << particle.position.x << " "
                    << particle.position.y << " "
                    << particle.position.z << "\n";
        }
        vtpFile << "        </DataArray>\n";
        vtpFile << "      </Points>\n";

        // Vertices section
        vtpFile << "      <Verts>\n";
        vtpFile << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t i = 0; i < particles.size(); ++i) {
            vtpFile << "          " << i << "\n";
        }
        vtpFile << "        </DataArray>\n";
        vtpFile << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t i = 1; i <= particles.size(); ++i) {
            vtpFile << "          " << i << "\n";
        }
        vtpFile << "        </DataArray>\n";
        vtpFile << "      </Verts>\n";

        // Point Data section
        vtpFile << "      <PointData>\n";

        // Radius attribute
        vtpFile << "        <DataArray type=\"Float32\" Name=\"Radius\" format=\"ascii\" NumberOfComponents=\"1\">\n";
        for (const auto& particle : particles) {
            vtpFile << "          " << particle.getRadius() << "\n";
        }
        vtpFile << "        </DataArray>\n";

        // Velocity attribute
        vtpFile << "        <DataArray type=\"Float32\" Name=\"Velocity\" format=\"ascii\" NumberOfComponents=\"3\">\n";
        for (const auto& particle : particles) {
            vtpFile << "          "
                    << particle.velocity.x << " "
                    << particle.velocity.y << " "
                    << particle.velocity.z << "\n";
        }
        vtpFile << "        </DataArray>\n";

        // Optional: Additional attributes can be added here
        // For example, force, angular velocity, etc.

        vtpFile << "      </PointData>\n";

        // Closing tags
        vtpFile << "    </Piece>\n";
        vtpFile << "  </PolyData>\n";
        vtpFile << "</VTKFile>\n";

        vtpFile.close();
}

void Output::writeTree(const QuadTree& rootNode, int timestep)
{
    // Get the tree configuration - you need to have access to this
    // You may need to add this as a member to the Output class or pass it as a parameter
    TreeConfig config; // Replace with actual configuration access

    std::string filename = createFilename("quadtree", timestep, ".vtu");

    // We need to pass the entire tree array, not just the root node
    const QuadTree* treeArray = &rootNode;

    // Get the maximum depth from your tree configuration
    // int maxDepth = config.maxDepth;

    QuadTreeWriter::writeQuadTree(filename, treeArray, config);
}