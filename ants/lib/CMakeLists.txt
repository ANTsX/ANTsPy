cmake_minimum_required(VERSION 3.0)

project(ants)

#set(PYBIND11_CPP_STANDARD -std=c++14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


## SETUP ITK ##
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


## SETUP PYBIND11 ##
add_subdirectory(pybind11)

## BUILD ANTS LIBRARIES ##
add_library(antsUtilities STATIC antscore/antsUtilities.cxx
                                 antscore/antsCommandLineOption.cxx
                                 antscore/antsCommandLineParser.cxx
                                 antscore/ReadWriteData.cxx
                                 antscore/ANTsVersion.cxx )

add_library(registrationUtilities STATIC  antscore/antsRegistrationTemplateHeader.cxx
                                          antscore/antsRegistration2DDouble.cxx antscore/antsRegistration2DFloat.cxx
                                          antscore/antsRegistration3DDouble.cxx antscore/antsRegistration3DFloat.cxx
                                          antscore/antsRegistration4DDouble.cxx antscore/antsRegistration4DFloat.cxx)

add_library(imageMathUtilities STATIC   antscore/ImageMathHelper2D.cxx
                                        antscore/ImageMathHelper3D.cxx  antscore/ImageMathHelper4D.cxx)
target_link_libraries(antsUtilities ${ITK_LIBRARIES})
target_link_libraries(registrationUtilities ${ITK_LIBRARIES})
target_link_libraries(imageMathUtilities ${ITK_LIBRARIES})

# ------------------------------------------------------------
## BUILD ANTSPY LIBRARY ##

# MODULES

## LOCAL ##
pybind11_add_module(addNoiseToImage LOCAL_addNoiseToImage.cxx)
pybind11_add_module(antiAlias LOCAL_antiAlias.cxx)
pybind11_add_module(antsImage LOCAL_antsImage.cxx)
pybind11_add_module(antsImageClone LOCAL_antsImageClone.cxx)
pybind11_add_module(antsImageHeaderInfo LOCAL_antsImageHeaderInfo.cxx)
pybind11_add_module(antsImageMutualInformation LOCAL_antsImageMutualInformation.cxx)
pybind11_add_module(antsImageToImageMetric LOCAL_antsImageToImageMetric.cxx)
pybind11_add_module(antsImageUtils LOCAL_antsImageUtils.cxx)
pybind11_add_module(antsTransform LOCAL_antsTransform.cxx)
pybind11_add_module(composeDisplacementFields LOCAL_composeDisplacementFields.cxx)
pybind11_add_module(cropImage LOCAL_cropImage.cxx)
pybind11_add_module(fitBsplineObjectToScatteredData LOCAL_fitBsplineObjectToScatteredData.cxx)
pybind11_add_module(fitBsplineDisplacementField LOCAL_fitBsplineDisplacementField.cxx)
pybind11_add_module(fitBsplineDisplacementFieldToScatteredData LOCAL_fitBsplineDisplacementFieldToScatteredData.cxx)
pybind11_add_module(fsl2antstransform LOCAL_fsl2antstransform.cxx)
pybind11_add_module(getNeighborhoodMatrix LOCAL_getNeighborhoodMatrix.cxx)
pybind11_add_module(hausdorffDistance LOCAL_hausdorffDistance.cxx)
pybind11_add_module(histogramMatchImage LOCAL_histogramMatchImages.cxx)
pybind11_add_module(integrateVelocityField LOCAL_integrateVelocityField.cxx)
pybind11_add_module(invertDisplacementField LOCAL_invertDisplacementField.cxx)
pybind11_add_module(labelOverlapMeasures LOCAL_labelOverlapMeasures.cxx)
pybind11_add_module(labelStats LOCAL_labelStats.cxx)
pybind11_add_module(mergeChannels LOCAL_mergeChannels.cxx)
pybind11_add_module(padImage LOCAL_padImage.cxx)
pybind11_add_module(readImage LOCAL_readImage.cxx)
pybind11_add_module(readTransform LOCAL_readTransform.cxx)
pybind11_add_module(reflectionMatrix LOCAL_reflectionMatrix.cxx)
pybind11_add_module(reorientImage LOCAL_reorientImage.cxx)
pybind11_add_module(reorientImage2 LOCAL_reorientImage2.cxx)
pybind11_add_module(rgbToVector LOCAL_rgbToVector.cxx)
pybind11_add_module(sccaner LOCAL_sccaner.cxx)
pybind11_add_module(simulateDisplacementField LOCAL_simulateDisplacementField.cxx)
pybind11_add_module(sliceImage LOCAL_sliceImage.cxx)
pybind11_add_module(SmoothImage LOCAL_SmoothImage.cxx)
pybind11_add_module(weingartenImageCurvature LOCAL_weingartenImageCurvature.cxx)

## WRAP ##
pybind11_add_module(antsAffineInitializer antscore/antsAffineInitializer.cxx WRAP_antsAffineInitializer.cxx)
pybind11_add_module(antsApplyTransforms antscore/antsApplyTransforms.cxx WRAP_antsApplyTransforms.cxx)
pybind11_add_module(antsApplyTransformsToPoints antscore/antsApplyTransformsToPoints.cxx WRAP_antsApplyTransformsToPoints.cxx)
pybind11_add_module(antsJointFusion antscore/antsJointFusion.cxx WRAP_antsJointFusion.cxx)
pybind11_add_module(antsRegistration antscore/antsRegistration.cxx WRAP_antsRegistration.cxx)
pybind11_add_module(Atropos antscore/Atropos.cxx WRAP_Atropos.cxx)
pybind11_add_module(CreateJacobianDeterminantImage antscore/CreateJacobianDeterminantImage.cxx WRAP_CreateJacobianDeterminantImage.cxx)
pybind11_add_module(CreateTiledMosaic antscore/CreateTiledMosaic.cxx WRAP_CreateTiledMosaic.cxx)
pybind11_add_module(DenoiseImage antscore/DenoiseImage.cxx WRAP_DenoiseImage.cxx)
pybind11_add_module(iMath antscore/iMath.cxx WRAP_iMath.cxx)
pybind11_add_module(KellyKapowski antscore/KellyKapowski.cxx WRAP_KellyKapowski.cxx)
pybind11_add_module(LabelClustersUniquely antscore/LabelClustersUniquely.cxx WRAP_LabelClustersUniquely.cxx)
pybind11_add_module(LabelGeometryMeasures antscore/LabelGeometryMeasures.cxx WRAP_LabelGeometryMeasures.cxx)
pybind11_add_module(N3BiasFieldCorrection antscore/N3BiasFieldCorrection.cxx WRAP_N3BiasFieldCorrection.cxx)
pybind11_add_module(N4BiasFieldCorrection antscore/N4BiasFieldCorrection.cxx WRAP_N4BiasFieldCorrection.cxx)
pybind11_add_module(ResampleImage antscore/ResampleImage.cxx WRAP_ResampleImage.cxx)
pybind11_add_module(ThresholdImage antscore/ThresholdImage.cxx WRAP_ThresholdImage.cxx)
pybind11_add_module(TileImages antscore/TileImages.cxx WRAP_TileImages.cxx)

## CONTRIB ##
pybind11_add_module(antsImageAugment CONTRIB_antsImageAugment.cxx)


## NOT-WRAPPED ##
#pybind11_add_module(antsLandmarkBasedTransformInitializer antscore/antsLandmarkBasedTransformInitializer.cxx WRAP_antsLandmarkBasedTransformInitializer.cxx)
#pybind11_add_module(antsMotionCorr antscore/antsMotionCorr.cxx WRAP_antsMotionCorr.cxx)
#pybind11_add_module(antsMotionCorrStats antscore/antsMotionCorrStats.cxx WRAP_antsMotionCorrStats.cxx)
#pybind11_add_module(antsSliceRegularizedRegistration antscore/antsSliceRegularizedRegistration.cxx WRAP_antsSliceRegularizedRegistration.cxx)
#pybind11_add_module(LesionFilling antscore/LesionFilling.cxx WRAP_LesionFilling.cxx)
#pybind11_add_module(NonLocalSuperResolution antscore/NonLocalSuperResolution.cxx WRAP_NonLocalSuperResolution.cxx)
#pybind11_add_module(SuperResolution antscore/SuperResolution.cxx WRAP_SuperResolution.cxx)
#pybind11_add_module(TimeSCCAN antscore/TimeSCCAN.cxx WRAP_TimeSCCAN.cxx)

# -----------------------------------------------------------------

## LOCAL ##
target_link_libraries(addNoiseToImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antiAlias PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageClone PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageHeaderInfo PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageMutualInformation PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageToImageMetric PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsImageUtils PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(antsTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(composeDisplacementFields PRIVATE ${ITK_LIBRARIES})
target_link_libraries(cropImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(fitBsplineObjectToScatteredData PRIVATE ${ITK_LIBRARIES})
target_link_libraries(fitBsplineDisplacementField PRIVATE ${ITK_LIBRARIES})
target_link_libraries(fitBsplineDisplacementFieldToScatteredData PRIVATE ${ITK_LIBRARIES})
target_link_libraries(fsl2antstransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(getNeighborhoodMatrix PRIVATE ${ITK_LIBRARIES})
target_link_libraries(hausdorffDistance PRIVATE ${ITK_LIBRARIES})
target_link_libraries(histogramMatchImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(integrateVelocityField PRIVATE ${ITK_LIBRARIES})
target_link_libraries(invertDisplacementField PRIVATE ${ITK_LIBRARIES})
target_link_libraries(labelOverlapMeasures PRIVATE ${ITK_LIBRARIES})
target_link_libraries(labelStats PRIVATE ${ITK_LIBRARIES})
target_link_libraries(mergeChannels PRIVATE ${ITK_LIBRARIES})
target_link_libraries(padImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(readImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(readTransform PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reflectionMatrix PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reorientImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(reorientImage2 PRIVATE ${ITK_LIBRARIES})
target_link_libraries(rgbToVector PRIVATE ${ITK_LIBRARIES})
target_link_libraries(sccaner PRIVATE ${ITK_LIBRARIES})
target_link_libraries(simulateDisplacementField PRIVATE ${ITK_LIBRARIES})
target_link_libraries(sliceImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(SmoothImage PRIVATE ${ITK_LIBRARIES})
target_link_libraries(weingartenImageCurvature PRIVATE ${ITK_LIBRARIES})

## WRAP ##
target_link_libraries(antsAffineInitializer PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
target_link_libraries(antsApplyTransforms PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
target_link_libraries(antsApplyTransformsToPoints PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
target_link_libraries(antsJointFusion PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
target_link_libraries(antsRegistration PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
target_link_libraries(Atropos PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(CreateJacobianDeterminantImage PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(CreateTiledMosaic PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(DenoiseImage PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(iMath PRIVATE ${ITK_LIBRARIES} antsUtilities imageMathUtilities)
target_link_libraries(KellyKapowski PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(LabelClustersUniquely PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(LabelGeometryMeasures PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(N3BiasFieldCorrection PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(N4BiasFieldCorrection PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(ResampleImage PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(ThresholdImage PRIVATE ${ITK_LIBRARIES} antsUtilities)
target_link_libraries(TileImages PRIVATE ${ITK_LIBRARIES} antsUtilities)

## CONTRIB ##
target_link_libraries(antsImageAugment PRIVATE ${ITK_LIBRARIES})


## NOT-WRAPPED ##
#target_link_libraries(antsLandmarkBasedTransformInitializer PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
#target_link_libraries(antsMotionCorr PRIVATE ${ITK_LIBRARIES} antsUtilities)
#target_link_libraries(antsMotionCorrStats PRIVATE ${ITK_LIBRARIES} antsUtilities)
#target_link_libraries(antsSliceRegularizedRegistration PRIVATE ${ITK_LIBRARIES} antsUtilities registrationUtilities)
#target_link_libraries(LesionFilling PRIVATE ${ITK_LIBRARIES} antsUtilities)
#target_link_libraries(NonLocalSuperResolution PRIVATE ${ITK_LIBRARIES} antsUtilities)
#target_link_libraries(SuperResolution PRIVATE ${ITK_LIBRARIES} antsUtilities)
#target_link_libraries(TimeSCCAN PRIVATE ${ITK_LIBRARIES} antsUtilities)

# -----------------------------------------------------------------
## VIZ ##
#pybind11_add_module(antsSurf antscore/antsSurf.cxx WRAP_antsSurf.cxx)
#pybind11_add_module(antsVol antscore/antsVol.cxx WRAP_antsVol.cxx)
#pybind11_add_module(ConvertScalarImageToRGB antscore/ConvertScalarImageToRGB.cxx WRAP_ConvertScalarImageToRGB.cxx)
#
#target_link_libraries(antsSurf PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES} antsUtilities)
#target_link_libraries(antsVol PRIVATE ${ITK_LIBRARIES} ${VTK_LIBRARIES} antsUtilities)
# target_link_libraries(ConvertScalarImageToRGB PRIVATE ${ITK_LIBRARIES} antsUtilities)
