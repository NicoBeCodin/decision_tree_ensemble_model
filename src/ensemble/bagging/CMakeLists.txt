add_library(Bagging bagging.cpp)

if (OPENMP)
    target_link_libraries(Bagging PRIVATE ${OPENMP_LIB})
endif()


# Include header file path
target_include_directories(Bagging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link dependencies
target_link_libraries(Bagging PUBLIC FunctionsTree)
target_link_libraries(Bagging PRIVATE LossFunction)
if (MPI_LIB)
    target_link_libraries(Bagging PRIVATE ${MPI_LIB})
endif()
