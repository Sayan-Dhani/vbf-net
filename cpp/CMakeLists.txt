# VBF-Net C++ inference library.
#
#   cmake -S cpp -B cpp/build && cmake --build cpp/build -j
#
# Options:
#   -DVBFNET_ARCH=x86-64-v3   value of -march on x86-64 (default: AVX2/FMA, any CPU
#                             since 2013). "native" = the build machine only;
#                             "" = the compiler's default (slower, runs anywhere).
#   -DVBFNET_BUILD_EXAMPLES=ON  build the examples (the ROOT ones only if ROOT is found)
cmake_minimum_required(VERSION 3.16)
project(vbfnet VERSION 5.0.0 LANGUAGES CXX)

option(VBFNET_BUILD_EXAMPLES "Build the example programs" ON)
set(VBFNET_ARCH "x86-64-v3" CACHE STRING "-march value on x86-64 (empty: compiler default)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)

add_library(vbfnet SHARED src/VBFNet.cc)
add_library(vbfnet::vbfnet ALIAS vbfnet)
target_include_directories(vbfnet PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(vbfnet PUBLIC cxx_std_17)
set_target_properties(vbfnet PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # FMA contraction where the CPU has it; results stay within float32 rounding.
  target_compile_options(vbfnet PRIVATE -Wall -Wextra -ffp-contract=fast)
  # Never -ffast-math: the features and the decode need IEEE semantics.
  if(VBFNET_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(vbfnet PRIVATE -march=${VBFNET_ARCH})
  endif()
endif()

if(VBFNET_BUILD_EXAMPLES)
  add_executable(vbfnet_minimal examples/minimal.cc)
  target_link_libraries(vbfnet_minimal PRIVATE vbfnet)

  find_package(ROOT QUIET COMPONENTS ROOTDataFrame)
  if(ROOT_FOUND)
    add_executable(vbfnet_predict_tree examples/predict_tree.cc)
    target_link_libraries(vbfnet_predict_tree PRIVATE vbfnet ROOT::ROOTDataFrame ROOT::Tree ROOT::Core)
    # The tool declares the header to ROOT's interpreter (for RDataFrame expressions).
    target_compile_definitions(vbfnet_predict_tree PRIVATE
      VBFNET_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include")
    message(STATUS "vbfnet: ROOT ${ROOT_VERSION} found, building vbfnet_predict_tree")
  else()
    message(STATUS "vbfnet: ROOT not found, vbfnet_predict_tree is not built")
  endif()
endif()

install(TARGETS vbfnet EXPORT vbfnetTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vbfnetTargets NAMESPACE vbfnet:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vbfnet)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vbfnetConfig.cmake
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/vbfnetTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vbfnetConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vbfnet)
if(TARGET vbfnet_predict_tree)
  install(TARGETS vbfnet_predict_tree RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
