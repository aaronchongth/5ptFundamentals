#-------------------------------------------------------------------------------
project(main C CXX)

set(SOURCES
    main.cpp
)

set(PRIVATE_LIBRARIES
    ${PRIVATE_LIBRARIES})

#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------
set(PRIVATE_INCLUDES ${OpenCV_INCLUDE_DIRS} ${PRIVATE_INCLUDES})
set(PRIVATE_INCLUDES ${cxxopts_INCLUDE_DIRS} ${PRIVATE_INCLUDES})
set(PRIVATE_INCLUDES ${eigen_INCLUDE_DIRS} ${PRIVATE_INCLUDES})

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
set(PRIVATE_LIBRARIES ${OpenCV_LIBS} ${PRIVATE_LIBRARIES})
set(PRIVATE_LIBRARIES utilities ${PRIVATE_LIBRARIES})
set(PRIVATE_LIBRARIES 7pt_algo ${PRIVATE_LIBRARIES})
set(PRIVATE_LIBRARIES 5pt_algo ${PRIVATE_LIBRARIES})

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      ${PRIVATE_LIBRARIES})
target_include_directories(${PROJECT_NAME}
                           PRIVATE
                           ${PRIVATE_INCLUDES})

