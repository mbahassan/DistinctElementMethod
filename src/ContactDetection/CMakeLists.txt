set(LIB_NAME ContactDetection)

set(SOURCES
        ContactDetection.cu
#        BroadPhase/Octree/OctreeBuilder.cu
#        BroadPhase/Octree/OctreeBuilderKernel.cuh

        BroadPhase/BroadPhase.cpp
        BroadPhase/QuadTree/QuadTreeBuilder.cu
        BroadPhase/QuadTree/QuadTreeBuilderKernel.cuh

        # Narrow phase detection
        NarrowPhase/NarrowPhase.cu
        NarrowPhase/EPA/EPA.cpp
        NarrowPhase/GJK/GJK.cpp
        NarrowPhase/Simplex/Simplex.cpp
)

# Enable CUDA dynamic parallelism
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")


add_library( ${LIB_NAME} SHARED  ${SOURCES})
install( TARGETS ${LIB_NAME} DESTINATION lib)

# Add these properties
set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${LIB_NAME}
        PUBLIC
        CUDA::cudart)

# Include headers from the current directory
target_include_directories(${LIB_NAME}
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}   # CUDA headers
        ${PROJECT_SOURCE_DIR}/src     # Current directory headers
)
