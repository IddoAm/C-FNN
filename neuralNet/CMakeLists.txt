cmake_minimum_required(VERSION 3.0)
project(CNerualNetwork)

# Add library target
add_library(mnist_parser STATIC mnist_parser.c)

# Set include directories for the library
target_include_directories(mnist_parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add executable target
add_executable(prgm main.c)

# Set include directories for the executable
target_include_directories(prgm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link the library with the executable
target_link_libraries(prgm PRIVATE mnist_parser)

# Link OpenBLAS library
find_package(OpenBLAS REQUIRED ARCH x86)
target_link_libraries(prgm PRIVATE OpenBLAS::OpenBLAS)

# Optionally, you can set C standard
# set_property(TARGET your_executable PROPERTY C_STANDARD 11)

# Optionally, you can add compiler flags
# target_compile_options(your_executable PRIVATE -Wall -Wextra -pedantic)

# Optionally, you can set output directory
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Optionally, you can set source group for IDE
# source_group("Source Files" FILES main.c mnist_parser.c)
# source_group("Header Files" FILES mnist_parser.h)
