set(LIB_NAME Tools)

set(SOURCES
#        Position/Position.cpp
        CudaHelper.cpp
        quaternion/quaternion.hpp

#        tinyobjloader/objloader.cpp
#        ObjReader/ObjReader.cpp
        StlReader/StlReader.cpp

)

add_library( ${LIB_NAME} SHARED  ${SOURCES})
install( TARGETS ${LIB_NAME} DESTINATION lib)

target_link_libraries(${LIB_NAME} CUDA::cudart)

# Include headers from the current directory
target_include_directories(${LIB_NAME}
        PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}   # CUDA headers
        ${PROJECT_SOURCE_DIR}/src     # Current directory headers
)