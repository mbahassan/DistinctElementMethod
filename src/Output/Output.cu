//
// Created by iqraa on 5-3-25.
//

#include "Output.cuh"

#include "QuadTreeWriter.cuh"

void Output::writeParticles(const std::vector<Spherical> &particles, const int timestep) {

    // Create filename with zero-padded timestep
    std::string filename = createFilename("Particles_Spherical", timestep, ".vtp");
    std::ofstream vtpFile(filename);

    if (!vtpFile.is_open())
    {
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


void Output::writeParticles(const std::vector<Polyhedral> &particles, const int timestep)
{
    // Create filename with zero-padded timestep
    std::string filename = createFilename("Particles_Poly", timestep, ".vtp");
    std::ofstream vtpFile(filename);

    if (!vtpFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Calculate total number of vertices and faces across all particles
    size_t totalVertices = 0;
    size_t totalFaces = 0;
    size_t totalIndices = 0;

    for (const auto& particle : particles) {
        totalVertices += particle.numVertices;
        totalFaces += particle.numTriangles ;

        // Count total indices by adding up the number of vertices in each face
        for (size_t i = 0; i < particle.numTriangles; i++)
        {
            // auto face = particle.getFaces()[i];
            totalIndices += 3;
        }
    }

    // XML Header
    vtpFile << "<?xml version=\"1.0\"?>\n";
    vtpFile << "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtpFile << "  <PolyData>\n";
    vtpFile << "    <Piece NumberOfPoints=\"" << totalVertices << "\" NumberOfPolys=\"" << totalFaces << "\">\n";

    // Points section
    vtpFile << "      <Points>\n";
    vtpFile << "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    // Write all transformed vertices
    size_t vertexOffset = 0;
    for (const auto& particle : particles)
    {
        const Quaternion& q = particle.orientation;
        // Precompute the quaternion conjugate (inverse for unit quaternions)
        Quaternion qConj = q.conjugate();

        for (int i = 0; i < particle.numVertices; i++)
        {

            float3 v = particle.vertices[i];

            // Create a quaternion from the vertex (with w=0)
            // Quaternion vQuat(0.0f, v.x, v.y, v.z);

            // Perform Hamilton product: q * v * q^(-1)
            // Quaternion rotatedVQuat = q.multiply(vQuat).multiply(qConj);
            // Quaternion rotatedVQuat = q*(vQuat)*(qConj);
            // Quaternion rotatedVQuat = rotateVector(v);

            // Extract the vector part of the resulting quaternion
            // float3 rotatedVertex = {rotatedVQuat.x, rotatedVQuat.y, rotatedVQuat.z};
            float3 rotatedVertex = q.rotateVector(v);

            // Apply translation
            float3 transformedVertex = rotatedVertex + particle.position;

            vtpFile << "          "
                    << transformedVertex.x << " "
                    << transformedVertex.y << " "
                    << transformedVertex.z << "\n";
        }
    }

    vtpFile << "        </DataArray>\n";
    vtpFile << "      </Points>\n";

    // Polys section (faces)
    vtpFile << "      <Polys>\n";

    // Connectivity array - lists all vertex indices for each face
    vtpFile << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    vertexOffset = 0;
    for (const auto& particle : particles) {
        for (int i = 0; i < particle.numTriangles; i++) {
            int3 triangle = particle.triangles[i];
            vtpFile << "          "
                    << (triangle.x + vertexOffset) << " "
                    << (triangle.y + vertexOffset) << " "
                    << (triangle.z + vertexOffset) << "\n";
        }
        vertexOffset += particle.numVertices ;
    }
    vtpFile << "        </DataArray>\n";

    // Offsets array - keeps track of where each face ends in connectivity
    vtpFile << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    size_t offset = 0;
    for (const auto& particle : particles)
    {
        for (size_t i = 0; i < particle.numTriangles; i++)
        {
            offset += 3;
            vtpFile << "          " << offset << "\n";
        }
    }
    vtpFile << "        </DataArray>\n";
    vtpFile << "      </Polys>\n";

    /// Point Data section
    vtpFile << "      <PointData>\n";

    // ParticleID attribute - helps identify which particle each vertex belongs to
    vtpFile << "        <DataArray type=\"Int32\" Name=\"VertexID\" format=\"ascii\" NumberOfComponents=\"1\">\n";
    for (const auto & particle : particles)
    {
        for (size_t i = 0; i < particle.numVertices; i++)
        {
            vtpFile << "          " << particle.getId() << "\n";
        }
    }
    vtpFile << "        </DataArray>\n";
    vtpFile << "      </PointData>\n";

    /// Cell Data section (data for each face)
    vtpFile << "      <CellData>\n";

    // ParticleID attribute for cells
    vtpFile << "        <DataArray type=\"Int32\" Name=\"CellsID\" format=\"ascii\" NumberOfComponents=\"1\">\n";
    for (const auto & particle : particles)
    {
        for (size_t i = 0; i < particle.numTriangles; i++)
        {
            vtpFile << "          " << particle.getId() << "\n";
        }
    }
    vtpFile << "        </DataArray>\n";

    // Velocity attribute
    vtpFile << "        <DataArray type=\"Float32\" Name=\"Velocity\" format=\"ascii\" NumberOfComponents=\"3\">\n";
    for (const auto& particle : particles)
    {
        for (size_t i = 0; i < particle.numTriangles; i++)
        {
            vtpFile << "          0 0 0\n";
        }
    }
    vtpFile << "        </DataArray>\n";

    // Angular velocity attribute
    vtpFile << "        <DataArray type=\"Float32\" Name=\"AngularVelocity\" format=\"ascii\" NumberOfComponents=\"3\">\n";
    for (const auto& particle : particles)
    {
        for (size_t i = 0; i < particle.numTriangles; i++)
        {
            vtpFile << "          0 0 0\n";
        }
    }

    vtpFile << "        </DataArray>\n";

    vtpFile << "      </CellData>\n";

    // Closing tags
    vtpFile << "    </Piece>\n";
    vtpFile << "  </PolyData>\n";
    vtpFile << "</VTKFile>\n";

    vtpFile.close();
}

void Output::writeTree(const QuadTree* quadtree, int timestep)
{
    // Get the tree configuration - you need to have access to this
    // You may need to add this as a member to the Output class or pass it as a parameter
    TreeConfig config(4,2); // Replace with actual configuration access
    // config.origin = {0,0,0};
    // config.size = {1,1,1};


    std::string filename = createFilename("Quadtree", timestep, ".vtu");

    // We need to pass the entire tree array, not just the root node
    const QuadTree* tree = quadtree;

    // Get the maximum depth from your tree configuration
    // int maxDepth = config.maxDepth;

    QuadTreeWriter::writeQuadTree(filename, tree, config);
}