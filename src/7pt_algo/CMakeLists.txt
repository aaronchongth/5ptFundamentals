#-------------------------------------------------------------------------------
project(7pt_algo C CXX)

set(SOURCES
    7pt.cpp
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
set(PRIVATE_LIBRARIES data_handler ${PRIVATE_LIBRARIES})

#-------------------------------------------------------------------------------
# Final library target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      ${PRIVATE_LIBRARIES})
target_include_directories(${PROJECT_NAME}
                           PRIVATE
                           ${PRIVATE_INCLUDES})

