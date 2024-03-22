cmake_minimum_required(VERSION 3.15...3.26)

project(ants LANGUAGES CXX)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.11
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

#execute_process(COMMAND sh ./scripts/configure_ITK.sh)
#execute_process(COMMAND sh ./scripts/configure_ANTs.sh)

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
  STABLE_ABI
  NB_STATIC
  src/main.cpp
  src/antscore/antsAffineInitializer.cxx
  src/antscore/Atropos.cxx
  src/antscore/antsRegistration.cxx
  src/antscore/iMath.cxx
  src/antscore/ThresholdImage.cxx
)
target_link_libraries(lib PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities imageMathUtilities)

# Install directive for scikit-build-core
install(TARGETS lib LIBRARY DESTINATION ants)

