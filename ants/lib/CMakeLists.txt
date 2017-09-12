cmake_minimum_required(VERSION 3.0)

project(ants)

#set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(VTK)

add_subdirectory(pybind11)

# ------------------------------------------------------------
# MODULES

## CORE
pybind11_add_module(readImage readImage.cxx)
pybind11_add_module(antsImage antsImage.cxx)
pybind11_add_module(readTransform readTransform.cxx)
pybind11_add_module(antsTransform antsTransform.cxx)
pybind11_add_module(antsImageHeaderInfo antsImageHeaderInfo.cxx)
pybind11_add_module(antsImageToImageMetric antsImageToImageMetric.cxx)

## SEGMENTATION
pybind11_add_module(Atropos Atropos.cxx)
pybind11_add_module(KellyKapowski KellyKapowski.cxx)
pybind11_add_module(antsJointFusion antsJointFusion.cxx)

## REGISTRATION
pybind11_add_module(antsRegistration antsRegistration.cxx)
pybind11_add_module(antsImageMutualInformation antsImageMutualInformation.cxx)
pybind11_add_module(antsApplyTransforms antsApplyTransforms.cxx)
pybind11_add_module(CreateJacobianDeterminantImage CreateJacobianDeterminantImage.cxx)
pybind11_add_module(ResampleImage ResampleImage.cxx)
pybind11_add_module(antsAffineInitializer antsAffineInitializer.cxx)

## LEARN
pybind11_add_module(sccaner sccaner.cxx)

## UTILS
pybind11_add_module(N3BiasFieldCorrection N3BiasFieldCorrection.cxx)
pybind11_add_module(N4BiasFieldCorrection N4BiasFieldCorrection.cxx)
pybind11_add_module(ThresholdImage ThresholdImage.cxx)
pybind11_add_module(iMath iMath.cxx)
pybind11_add_module(LabelClustersUniquely LabelClustersUniquely.cxx)
pybind11_add_module(SmoothImage SmoothImage.cxx)
pybind11_add_module(reflectionMatrix reflectionMatrix.cxx)
pybind11_add_module(cropImage cropImage.cxx)
pybind11_add_module(mergeChannels mergeChannels.cxx)
pybind11_add_module(DenoiseImage DenoiseImage.cxx)
pybind11_add_module(reorientImage reorientImage.cxx)
pybind11_add_module(weingartenImageCurvature weingartenImageCurvature.cxx)
pybind11_add_module(labelStats labelStats.cxx)

# ------------------------------------------------------------
# LINKING

## CORE
target_link_libraries(readImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(readTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageHeaderInfo PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageToImageMetric PRIVATE ${ITK_LIBRARIES})

## SEGMENTATION
target_link_libraries(Atropos PRIVATE ${ITK_LIBRARIES})
target_link_libraries(KellyKapowski PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsJointFusion PRIVATE ${ITK_LIBRARIES})

## REGISTRATION
target_link_libraries(antsRegistration PRIVATE ${ITK_LIBRARIES} ${ITKIOImageBase_LIBRARIES} ${ITKZLIB_LIBRARIES})
target_link_libraries(antsImageMutualInformation PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsApplyTransforms PRIVATE ${ITK_LIBRARIES})
target_link_libraries(CreateJacobianDeterminantImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(ResampleImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsAffineInitializer PRIVATE ${ITK_LIBRARIES})

## LEARN
target_link_libraries(sccaner PRIVATE ${ITK_LIBRARIES})

## UTILS
target_link_libraries(N3BiasFieldCorrection PRIVATE ${ITK_LIBRARIES})
target_link_libraries(N4BiasFieldCorrection PRIVATE ${ITK_LIBRARIES})
target_link_libraries(ThresholdImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(iMath PRIVATE ${ITK_LIBRARIES})
target_link_libraries(LabelClustersUniquely PRIVATE ${ITK_LIBRARIES})
target_link_libraries(SmoothImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reflectionMatrix PRIVATE ${ITK_LIBRARIES})
target_link_libraries(cropImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(mergeChannels PRIVATE ${ITK_LIBRARIES})
target_link_libraries(DenoiseImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reorientImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(weingartenImageCurvature PRIVATE ${ITK_LIBRARIES})
target_link_libraries(labelStats PRIVATE ${ITK_LIBRARIES})


if(NOT VTK)
    message("VTK library not found... not building ants.Surf or ants.Vol")
else()
    include(${VTK_USE_FILE})
    pybind11_add_module(antsSurf antsSurf.cxx)
    pybind11_add_module(antsVol antsVol.cxx)
    pybind11_add_module(ConvertScalarImageToRGB ConvertScalarImageToRGB.cxx)
    target_link_libraries(antsSurf PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES})
    target_link_libraries(antsVol PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES})
    target_link_libraries(ConvertScalarImageToRGB PRIVATE ${ITK_LIBRARIES})
endif()

