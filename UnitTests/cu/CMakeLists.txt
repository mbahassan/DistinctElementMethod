file(GLOB CPPFILES  "*.cu")
#for every cpp found
foreach(CPPFILE ${CPPFILES})
    #extract the actually file name
    get_filename_component(FILENAME ${CPPFILE} NAME)
    #extract the filename minus the cpp. This will be the name of exe file
    get_filename_component(EXECNAME ${CPPFILE} NAME_WE)
    #Make the exe
    add_executable(${EXECNAME} ${FILENAME})
    #All cpp folder and linked against MercuryBase
    target_link_libraries(${EXECNAME}  PUBLIC Particle Tools)
endforeach()