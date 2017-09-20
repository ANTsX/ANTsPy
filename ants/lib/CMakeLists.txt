cmake_minimum_required(VERSION 3.0)

project(ants)

#set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(PYBIND11_CPP_STANDARD -std=c++14)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(VTK)

add_subdirectory(pybind11)

# ------------------------------------------------------------
# MODULES

## LOCAL ##
pybind11_add_module(antsImage LOCAL_antsImage.cxx)
pybind11_add_module(antsImageClone LOCAL_antsImageClone.cxx)
pybind11_add_module(antsImageHeaderInfo LOCAL_antsImageHeaderInfo.cxx)
pybind11_add_module(antsImageMutualInformation LOCAL_antsImageMutualInformation.cxx)
pybind11_add_module(antsImageToImageMetric LOCAL_antsImageToImageMetric.cxx)
pybind11_add_module(antsImageUtils LOCAL_antsImageUtils.cxx)
pybind11_add_module(antsTransform LOCAL_antsTransform.cxx)
pybind11_add_module(cropImage LOCAL_cropImage.cxx)
pybind11_add_module(fsl2antstransform LOCAL_fsl2antstransform.cxx)
pybind11_add_module(getNeighborhoodMatrix LOCAL_getNeighborhoodMatrix.cxx)
pybind11_add_module(invariantImageSimilarity LOCAL_invariantImageSimilarity.cxx)
pybind11_add_module(labelStats LOCAL_labelStats.cxx)
pybind11_add_module(mergeChannels LOCAL_mergeChannels.cxx)
pybind11_add_module(readImage LOCAL_readImage.cxx)
pybind11_add_module(readTransform LOCAL_readTransform.cxx)
pybind11_add_module(reflectionMatrix LOCAL_reflectionMatrix.cxx)
pybind11_add_module(reorientImage LOCAL_reorientImage.cxx)
pybind11_add_module(sccaner LOCAL_sccaner.cxx)
pybind11_add_module(SmoothImage LOCAL_SmoothImage.cxx)
pybind11_add_module(weingartenImageCurvature LOCAL_weingartenImageCurvature.cxx)

## WRAP ##
pybind11_add_module(antsAffineInitializer antscore/antsAffineInitializer.cxx WRAP_antsAffineInitializer.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx)
pybind11_add_module(antsApplyTransforms antscore/antsApplyTransforms.cxx WRAP_antsApplyTransforms.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsJointFusion antscore/antsJointFusion.cxx WRAP_antsJointFusion.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx antscore/ANTsVersion.cxx)
pybind11_add_module(antsLandmarkBasedTransformInitializer antscore/antsLandmarkBasedTransformInitializer.cxx WRAP_antsLandmarkBasedTransformInitializer.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsMotionCorr antscore/antsMotionCorr.cxx WRAP_antsMotionCorr.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsMotionCorrStats antscore/antsMotionCorrStats.cxx WRAP_antsMotionCorrStats.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsSliceRegularizedRegistration antscore/antsSliceRegularizedRegistration.cxx WRAP_antsSliceRegularizedRegistration.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx)
pybind11_add_module(antsLandmarkBasedTransformInitializer antscore/antsLandmarkBasedTransformInitializer.cxx WRAP_antsLandmarkBasedTransformInitializer.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsMotionCorr antscore/antsMotionCorr.cxx WRAP_antsMotionCorr.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(antsMotionCorrStats antscore/antsMotionCorrStats.cxx WRAP_antsMotionCorrStats.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
pybind11_add_module(Atropos antscore/Atropos.cxx WRAP_Atropos.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)

# -----------------------------------------------------------------
## LOCAL ##
target_link_libraries(antsImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageClone PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageHeaderInfo PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageMutualInformation PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageToImageMetric PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageUtils PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(cropImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(fsl2antstransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(getNeighborhoodMatrix PRIVATE ${ITK_LIBRARIES})
target_link_libraries(invariantImageSimilarity PRIVATE ${ITK_LIBRARIES})
target_link_libraries(labelStats PRIVATE ${ITK_LIBRARIES})
target_link_libraries(mergeChannels PRIVATE ${ITK_LIBRARIES})
target_link_libraries(readImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(readTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reflectionMatrix PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reorientImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(sccaner PRIVATE ${ITK_LIBRARIES})
target_link_libraries(SmoothImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(weingartenImageCurvature PRIVATE ${ITK_LIBRARIES})


## WRAP ##
target_link_libraries(antsAffineInitializer PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsApplyTransforms PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsJointFusion PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsLandmarkBasedTransformInitializer PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsMotionCorr PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsMotionCorrStats PRIVATE ${ITK_LIBRARIES})
target_link_libraries(Atropos PRIVATE ${ITK_LIBRARIES})

# -----------------------------------------------------------------
## VIZ ##
#pybind11_add_module(antsSurf antscore/antsSurf.cxx WRAP_antsSurf.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx)
#pybind11_add_module(antsVol antscore/antsVol.cxx WRAP_antsVol.cxx antscore/antsUtilities.cxx antscore/ReadWriteData.cxx antscore/antsCommandLineOption.cxx antscore/antsCommandLineParser.cxx antscore/ANTsVersion.cxx)
target_link_libraries(antsSurf PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsVol PRIVATE ${ITK_LIBRARIES})

