#include <nanobind/nanobind.h>

#include "addNoiseToImage.cxx"
#include "antiAlias.cxx"
#include "antsGetItem.cxx"
#include "antsImage.cxx"
#include "antsImageClone.cxx"
#include "antsImageHeaderInfo.cxx"
#include "antsImageMutualInformation.cxx"
#include "antsImageToImageMetric.cxx"
#include "antsImageUtils.cxx"
#include "antsTransform.cxx"
#include "composeDisplacementFields.cxx"
#include "cropImage.cxx"
#include "fitBsplineDisplacementFieldToScatteredData.cxx"
#include "fitBsplineObjectToScatteredData.cxx"
#include "fitThinPlateSplineDisplacementFieldToScatteredData.cxx"
#include "fsl2antstransform.cxx"
#include "getNeighborhoodMatrix.cxx"
#include "hausdorffDistance.cxx"
#include "hessianObjectness.cxx"
#include "histogramMatchImages.cxx"
#include "integrateVelocityField.cxx"
#include "invertDisplacementField.cxx"
#include "labelOverlapMeasures.cxx"
#include "labelStats.cxx"
#include "mergeChannels.cxx"
#include "padImage.cxx"
#include "readImage.cxx"
#include "readTransform.cxx"
#include "reflectionMatrix.cxx"
#include "reorientImage.cxx"
#include "reorientImage2.cxx"
#include "rgbToVector.cxx"
#include "sccaner.cxx"
#include "simulateDisplacementField.cxx"
#include "sliceImage.cxx"
#include "SmoothImage.cxx"
#include "weingartenImageCurvature.cxx"

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
void local_antiAlias(nb::module_ &);
void local_antsGetItem(nb::module_ &);
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
void local_hausdorffDistance(nb::module_ &);
void local_hessianObjectness(nb::module_ &);
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
    local_antiAlias(m);
    local_antsGetItem(m);
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
    local_hausdorffDistance(m);
    local_hessianObjectness(m);
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