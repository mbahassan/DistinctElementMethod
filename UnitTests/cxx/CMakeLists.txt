set(EXE_NAME  main)

set(SOURCES
        main.cpp
        GpuClass.cu
)

#Make the exe
add_executable(${EXE_NAME} ${SOURCES})

set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include headers from the current directory
target_include_directories(${EXE_NAME}
        PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}   # CUDA headers
)

#All cpp folder and linked a
target_link_libraries(${EXE_NAME}
        PUBLIC
        Base
        ContactDetection
        Insertion
        Output
        Particle
        TimeIntegrator
        Tools
)

