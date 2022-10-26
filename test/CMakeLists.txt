project(TEST)

add_subdirectory(lib)

include_directories(lib/googletest lib/googletest/include ../MetaAI)

file(GLOB_RECURSE tests ./*.cpp)

add_executable(TEST ${tests})

target_link_libraries(TEST gtest gtest_main)
target_link_libraries(TEST MetaAILib)