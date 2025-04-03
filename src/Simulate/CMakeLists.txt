set(LIB_NAME Simulation)

set(SOURCES
        Simulate.cu

        # Device Base Class
        Base/Base.cu
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
        Insertion # Particle is linked to it
        TimeIntegrator #
        Output
        CUDA::cudart)

# Include headers from the current directory
target_include_directories(${LIB_NAME}
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}   # CUDA headers
        ${PROJECT_SOURCE_DIR}/src     # Current directory headers
)
