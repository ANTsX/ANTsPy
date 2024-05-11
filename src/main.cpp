#include <nanobind/nanobind.h>

#include "LOCAL_addNoiseToImage.cxx"
#include "LOCAL_antsImage.cxx"
#include "LOCAL_antsImageClone.cxx"
#include "LOCAL_antsImageHeaderInfo.cxx"
#include "LOCAL_antsImageMutualInformation.cxx"
#include "LOCAL_antsImageToImageMetric.cxx"
#include "LOCAL_antsImageUtils.cxx"
#include "LOCAL_antsTransform.cxx"
#include "LOCAL_composeDisplacementFields.cxx"
#include "LOCAL_cropImage.cxx"
#include "LOCAL_fitBsplineDisplacementFieldToScatteredData.cxx"
#include "LOCAL_fitBsplineObjectToScatteredData.cxx"
#include "LOCAL_fitThinPlateSplineDisplacementFieldToScatteredData.cxx"
#include "LOCAL_fsl2antstransform.cxx"
#include "LOCAL_getNeighborhoodMatrix.cxx"
#include "LOCAL_histogramMatchImages.cxx"
#include "LOCAL_integrateVelocityField.cxx"
#include "LOCAL_invertDisplacementField.cxx"
#include "LOCAL_labelOverlapMeasures.cxx"
#include "LOCAL_labelStats.cxx"
#include "LOCAL_mergeChannels.cxx"
#include "LOCAL_padImage.cxx"
#include "LOCAL_readImage.cxx"
#include "LOCAL_readTransform.cxx"
#include "LOCAL_reflectionMatrix.cxx"
#include "LOCAL_reorientImage.cxx"
#include "LOCAL_reorientImage2.cxx"
#include "LOCAL_rgbToVector.cxx"
#include "LOCAL_sccaner.cxx"
#include "LOCAL_simulateDisplacementField.cxx"
#include "LOCAL_sliceImage.cxx"
#include "LOCAL_SmoothImage.cxx"
#include "LOCAL_weingartenImageCurvature.cxx"

#include "WRAP_antsAffineInitializer.cxx"
#include "WRAP_antsApplyTransforms.cxx"
#include "WRAP_antsApplyTransformsToPoints.cxx"
#include "WRAP_antsJointFusion.cxx"
#include "WRAP_antsRegistration.cxx"
#include "WRAP_Atropos.cxx"
#include "WRAP_AverageAffineTransform.cxx"
#include "WRAP_AverageAffineTransformNoRigid.cxx"
#include "WRAP_CreateJacobianDeterminantImage.cxx"
#include "WRAP_DenoiseImage.cxx"
#include "WRAP_iMath.cxx"
#include "WRAP_KellyKapowski.cxx"
#include "WRAP_LabelClustersUniquely.cxx"
#include "WRAP_LabelGeometryMeasures.cxx"
#include "WRAP_N3BiasFieldCorrection.cxx"
#include "WRAP_N4BiasFieldCorrection.cxx"
#include "WRAP_ResampleImage.cxx"
#include "WRAP_ThresholdImage.cxx"
#include "WRAP_TileImages.cxx"

namespace nb = nanobind;

void local_addNoiseToImage(nb::module_ &);
void local_antsImage(nb::module_ &);
void local_antsImageClone(nb::module_ &);
void local_antsImageHeaderInfo(nb::module_ &);
void local_antsImageMutualInformation(nb::module_ &);
void local_antsImageToImageMetric(nb::module_ &);
void local_antsImageUtils(nb::module_ &);
void local_antsTransform(nb::module_ &);
void local_cropImage(nb::module_ &);
void local_composeDisplacementFields(nb::module_ &);
void local_fitBsplineDisplacementFieldToScatteredData(nb::module_ &);
void local_fitBsplineObjectToScatteredData(nb::module_ &);
void local_fitThinPlateSplineDisplacementFieldToScatteredData(nb::module_ &);
void local_fsl2antstransform(nb::module_ &);
void local_getNeighborhoodMatrix(nb::module_ &);
void local_histogramMatchImages(nb::module_ &);
void local_integrateVelocityField(nb::module_ &);
void local_invertDisplacementField(nb::module_ &);
void local_labelOverlapMeasures(nb::module_ &);
void local_labelStats(nb::module_ &);
void local_mergeChannels(nb::module_ &);
void local_padImage(nb::module_ &);
void local_readImage(nb::module_ &);
void local_readTransform(nb::module_ &);
void local_reflectionMatrix(nb::module_ &);
void local_reorientImage(nb::module_ &);
void local_reorientImage2(nb::module_ &);
void local_rgbToVector(nb::module_ &);
void local_sccaner(nb::module_ &);
void local_simulateDisplacementField(nb::module_ &);
void local_sliceImage(nb::module_ &);
void local_SmoothImage(nb::module_ &);
void local_weingartenImageCurvature(nb::module_ &);

void wrap_antsAffineInitializer(nb::module_ &);
void wrap_antsApplyTransforms(nb::module_ &);
void wrap_antsApplyTransformsToPoints(nb::module_ &);
void wrap_antsJointFusion(nb::module_ &);
void wrap_antsRegistration(nb::module_ &);
void wrap_Atropos(nb::module_ &);
void wrap_AverageAffineTransform(nb::module_ &);
void wrap_AverageAffineTransformNoRigid(nb::module_ &);
void wrap_CreateJacobianDeterminantImage(nb::module_ &);
void wrap_DenoiseImage(nb::module_ &);
void wrap_iMath(nb::module_ &);
void wrap_KellyKapowski(nb::module_ &);
void wrap_LabelClustersUniquely(nb::module_ &);
void wrap_LabelGeometryMeasures(nb::module_ &);
void wrap_N3BiasFieldCorrection(nb::module_ &);
void wrap_N4BiasFieldCorrection(nb::module_ &);
void wrap_ResampleImage(nb::module_ &);
void wrap_ThresholdImage(nb::module_ &);
void wrap_TileImages(nb::module_ &);

NB_MODULE(lib, m) {
    local_addNoiseToImage(m);
    local_antsImage(m);
    local_antsImageClone(m);
    local_antsImageHeaderInfo(m);
    local_antsImageMutualInformation(m);
    local_antsImageToImageMetric(m);
    local_antsImageUtils(m);
    local_antsTransform(m);
    local_composeDisplacementFields(m);
    local_cropImage(m);
    local_fitBsplineDisplacementFieldToScatteredData(m);
    local_fitBsplineObjectToScatteredData(m);
    local_fitThinPlateSplineDisplacementFieldToScatteredData(m);
    local_fsl2antstransform(m);
    local_getNeighborhoodMatrix(m);
    local_histogramMatchImages(m);
    local_integrateVelocityField(m);
    local_invertDisplacementField(m);
    local_labelOverlapMeasures(m);
    local_labelStats(m);
    local_mergeChannels(m);
    local_padImage(m);
    local_readImage(m);
    local_readTransform(m);
    local_reflectionMatrix(m);
    local_reorientImage(m);
    local_reorientImage2(m);
    local_rgbToVector(m);
    local_sccaner(m);
    local_simulateDisplacementField(m);
    local_sliceImage(m);
    local_SmoothImage(m);
    local_weingartenImageCurvature(m);

    wrap_antsAffineInitializer(m);
    wrap_antsApplyTransforms(m);
    wrap_antsApplyTransformsToPoints(m);
    wrap_antsJointFusion(m);
    wrap_antsRegistration(m);
    wrap_Atropos(m);
    wrap_AverageAffineTransform(m);
    wrap_AverageAffineTransformNoRigid(m);
    wrap_CreateJacobianDeterminantImage(m);
    wrap_DenoiseImage(m);
    wrap_iMath(m);
    wrap_KellyKapowski(m);
    wrap_LabelClustersUniquely(m);
    wrap_LabelGeometryMeasures(m);
    wrap_N3BiasFieldCorrection(m);
    wrap_N4BiasFieldCorrection(m);
    wrap_ResampleImage(m);
    wrap_ThresholdImage(m);
    wrap_TileImages(m);
}