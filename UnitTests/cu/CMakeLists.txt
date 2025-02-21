set(EXECNAME  mainCu)

set(SOURCES
        mainCu.cu
)

#Make the exe
add_executable(${EXECNAME} ${SOURCES})

set_target_properties(${EXECNAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#All cpp folder and linked against MercuryBase
target_link_libraries(${EXECNAME}  PUBLIC Particle Tools)

target_include_directories(${EXECNAME} PUBLIC
)

target_link_libraries(${EXECNAME} PUBLIC
)