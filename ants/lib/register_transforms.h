#ifndef ANTS_REGISTER_TRANSFORM_H_
#define ANTS_REGISTER_TRANSFORM_H_

#include "itkTransform.h"
#include "itkTransformFactory.h"

void register_transforms()
{
    using MatrixOffsetTransformTypeA = itk::MatrixOffsetTransformBase<double, 3, 3>;
    itk::TransformFactory<MatrixOffsetTransformTypeA>::RegisterTransform();

    using MatrixOffsetTransformTypeB = itk::MatrixOffsetTransformBase<float, 3, 3>;
    itk::TransformFactory<MatrixOffsetTransformTypeB>::RegisterTransform();

    using MatrixOffsetTransformTypeC = itk::MatrixOffsetTransformBase<double, 2, 2>;
    itk::TransformFactory<MatrixOffsetTransformTypeC>::RegisterTransform();

    using MatrixOffsetTransformTypeD = itk::MatrixOffsetTransformBase<float, 2, 2>;
    itk::TransformFactory<MatrixOffsetTransformTypeD>::RegisterTransform();
}

#endif
