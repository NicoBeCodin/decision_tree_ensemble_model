# src/ensemble/boosting_advanced/CMakeLists.txt
add_library(Boosting_Advanced
    boosting_improved.cpp)

target_include_directories(Boosting_Advanced
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# OpenMP configuration
find_package(OpenMP REQUIRED)
target_link_libraries(Boosting_Advanced PRIVATE OpenMP::OpenMP_CXX)

# Headers for other targets to use
set_target_properties(Boosting_Advanced PROPERTIES
    PUBLIC_HEADER "boosting_advanced.h;boosting_improved.h;binning_methods.h")