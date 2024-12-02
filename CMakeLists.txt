cmake_minimum_required(VERSION 3.16.3...3.26)

project(ants LANGUAGES CXX)

# Set the macOS SDK root
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT ${SDK_PATH})
    message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# TODO: make this run only if ITK + ANTs are not already built
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  execute_process(COMMAND cmd /c ${PROJECT_SOURCE_DIR}/scripts/configure_ITK.bat)
  execute_process(COMMAND cmd /c ${PROJECT_SOURCE_DIR}/scripts/configure_ANTs.bat)
else()
  execute_process(COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/configure_ITK.sh)
  execute_process(COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/configure_ANTs.sh)
endif()

# ITK
set(ITK_DIR "./itkbuild")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# ANTS
add_library(antsUtilities STATIC src/antscore/antsUtilities.cxx src/antscore/antsCommandLineOption.cxx src/antscore/antsCommandLineParser.cxx src/antscore/ReadWriteData.cxx src/antscore/ANTsVersion.cxx)
add_library(registrationUtilities STATIC  src/antscore/antsRegistrationTemplateHeader.cxx
                                          src/antscore/antsRegistration2DDouble.cxx src/antscore/antsRegistration2DFloat.cxx
                                          src/antscore/antsRegistration3DDouble.cxx src/antscore/antsRegistration3DFloat.cxx
                                          src/antscore/antsRegistration4DDouble.cxx src/antscore/antsRegistration4DFloat.cxx)


add_library(imageMathUtilities STATIC  src/antscore/ImageMathHelper2D.cxx src/antscore/ImageMathHelper3D.cxx src/antscore/ImageMathHelper4D.cxx)

# this may not be needed
target_link_libraries(antsUtilities ${ITK_LIBRARIES})
target_link_libraries(registrationUtilities ${ITK_LIBRARIES})
target_link_libraries(imageMathUtilities ${ITK_LIBRARIES})

nanobind_add_module(
  lib
  NB_STATIC
  src/main.cpp
  src/antscore/antsAffineInitializer.cxx
  src/antscore/antsApplyTransforms.cxx
  src/antscore/antsApplyTransformsToPoints.cxx
  src/antscore/antsJointFusion.cxx
  src/antscore/antsRegistration.cxx
  src/antscore/Atropos.cxx
  src/antscore/AverageAffineTransform.cxx
  src/antscore/AverageAffineTransformNoRigid.cxx
  src/antscore/CreateJacobianDeterminantImage.cxx
  src/antscore/DenoiseImage.cxx
  src/antscore/iMath.cxx
  src/antscore/KellyKapowski.cxx
  src/antscore/LabelClustersUniquely.cxx
  src/antscore/LabelGeometryMeasures.cxx
  src/antscore/N3BiasFieldCorrection.cxx
  src/antscore/N4BiasFieldCorrection.cxx
  src/antscore/ResampleImage.cxx
  src/antscore/ThresholdImage.cxx
  src/antscore/TileImages.cxx
)
target_compile_options(lib PRIVATE -O3)
target_link_libraries(lib PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities imageMathUtilities)

# Install directive for scikit-build-core
install(TARGETS lib LIBRARY DESTINATION ants)