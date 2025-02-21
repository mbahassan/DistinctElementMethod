set(LIB_NAME Particle)

set(SOURCES
        Particle.cpp
        Material/Material.cpp
        Material/Material.hpp
        Shape/Shape.cpp
        Shape/Sphere/Sphere.cpp
)

add_library( ${LIB_NAME} SHARED  ${SOURCES})
install( TARGETS ${LIB_NAME} DESTINATION lib)

target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart)

# Include headers from the current directory
target_include_directories(${LIB_NAME}
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}   # CUDA headers
        ${PROJECT_SOURCE_DIR}/src     # Current directory headers
)
