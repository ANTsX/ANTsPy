#include <nanobind/nanobind.h>

#include "local_addNoiseToImage.cxx"
#include "local_antsImage.h"
#include "local_antsImage.cpp"
#include "local_antsImageClone.cpp"
#include "local_antsImageHeaderInfo.cpp"
#include "local_test.cpp"
#include "local_imageRead.cpp"
#include "WRAP_antsAffineInitializer.cxx"
#include "WRAP_antsApplyTransforms.cxx"
#include "WRAP_antsApplyTransformsToPoints.cxx"
#include "WRAP_antsJointFusion.cxx"
#include "WRAP_antsLandmarkBasedTransformInitializer.cxx"
#include "WRAP_antsMotionCorr.cxx"
#include "WRAP_antsMotionCorrStats.cxx"
#include "WRAP_antsRegistration.cxx"
#include "WRAP_antsSliceRegularizedRegistration.cxx"
#include "WRAP_Atropos.cxx"
#include "WRAP_AverageAffineTransform.cxx"
#include "WRAP_AverageAffineTransformNoRigid.cxx"
#include "WRAP_ConvertScalarImageToRGB.cxx"
#include "WRAP_CreateJacobianDeterminantImage.cxx"
#include "WRAP_DenoiseImage.cxx"
#include "WRAP_iMath.cxx"
#include "WRAP_KellyKapowski.cxx"
#include "WRAP_LabelClustersUniquely.cxx"
#include "WRAP_LabelGeometryMeasures.cxx"
#include "WRAP_LesionFilling.cxx"
#include "WRAP_N3BiasFieldCorrection.cxx"
#include "WRAP_N4BiasFieldCorrection.cxx"
#include "WRAP_ResampleImage.cxx"
#include "WRAP_SuperResolution.cxx"
#include "WRAP_ThresholdImage.cxx"
#include "WRAP_TileImages.cxx"

namespace nb = nanobind;


void local_addNoiseToImage(nb::module_ &);
void local_antsImage(nb::module_ &);
void local_antsImageClone(nb::module_ &);
void local_antsImageHeaderInfo(nb::module_ &);
void local_test(nb::module_ &);
void local_imageRead(nb::module_ &);

void wrap_antsAffineInitializer(nb::module_ &);
void wrap_antsApplyTransforms(nb::module_ &);
void wrap_antsApplyTransformsToPoints(nb::module_ &);
void wrap_antsJointFusion(nb::module_ &);
void wrap_antsLandmarkBasedTransformInitializer(nb::module_ &);
void wrap_antsMotionCorr(nb::module_ &);
void wrap_antsMotionCorrStats(nb::module_ &);
void wrap_antsRegistration(nb::module_ &);
void wrap_antsSliceRegularizedRegistration(nb::module_ &);
void wrap_Atropos(nb::module_ &);
void wrap_AverageAffineTransform(nb::module_ &);
void wrap_AverageAffineTransformNoRigid(nb::module_ &);
void wrap_ConvertScalarImageToRGB(nb::module_ &);
void wrap_CreateJacobianDeterminantImage(nb::module_ &);
void wrap_DenoiseImage(nb::module_ &);
void wrap_iMath(nb::module_ &);
void wrap_KellyKapowski(nb::module_ &);
void wrap_LabelClustersUniquely(nb::module_ &);
void wrap_LabelGeometryMeasures(nb::module_ &);
void wrap_LesionFilling(nb::module_ &);
void wrap_N3BiasFieldCorrection(nb::module_ &);
void wrap_N4BiasFieldCorrection(nb::module_ &);
void wrap_ResampleImage(nb::module_ &);
void wrap_SuperResolution(nb::module_ &);
void wrap_ThresholdImage(nb::module_ &);
void wrap_TileImages(nb::module_ &);

NB_MODULE(lib, m) {
    local_addNoiseToImage(m);
    local_antsImage(m);
    local_antsImageClone(m);
    local_antsImageHeaderInfo(m);
    local_test(m);
    local_imageRead(m);
    wrap_antsAffineInitializer(m);
    wrap_antsApplyTransforms(m);
    wrap_antsApplyTransformsToPoints(m);
    wrap_antsJointFusion(m);
    wrap_antsLandmarkBasedTransformInitializer(m);
    wrap_antsMotionCorr(m);
    wrap_antsMotionCorrStats(m);
    wrap_antsRegistration(m);
    wrap_antsSliceRegularizedRegistration(m);
    wrap_Atropos(m);
    wrap_AverageAffineTransform(m);
    wrap_AverageAffineTransformNoRigid(m);
    wrap_ConvertScalarImageToRGB(m);
    wrap_CreateJacobianDeterminantImage(m);
    wrap_DenoiseImage(m);
    wrap_iMath(m);
    wrap_KellyKapowski(m);
    wrap_LabelClustersUniquely(m);
    wrap_LabelGeometryMeasures(m);
    wrap_LesionFilling(m);
    wrap_N3BiasFieldCorrection(m);
    wrap_N4BiasFieldCorrection(m);
    wrap_ResampleImage(m);
    wrap_SuperResolution(m);
    wrap_ThresholdImage(m);
    wrap_TileImages(m);
}