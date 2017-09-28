#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_vector_ref.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"

#include "antscore/antsUtilities.h"

#include "itkAffineTransform.h"
#include "itkCenteredAffineTransform.h"
#include "itkEuler2DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkRigid2DTransform.h"
#include "itkRigid3DTransform.h"
#include "itkCenteredRigid2DTransform.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkCenteredSimilarity2DTransform.h"
#include "itkSimilarity3DTransform.h"
#include "itkQuaternionRigidTransform.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileReader.h"
#include "itkCompositeTransform.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkDisplacementFieldTransform.h"
#include "itkConstantBoundaryCondition.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkGaussianInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkWindowedSincInterpolateImageFunction.h"
#include "itkLabelImageGaussianInterpolateImageFunction.h"
#include "itkTransformFileWriter.h"

#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_vector_ref.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"

#include "LOCAL_readTransform.h"

namespace py = pybind11;


unsigned int  getTransformFileDimension( std::string filename )
{
    typedef itk::TransformFileReader TransformReaderType1;
    typedef typename TransformReaderType1::Pointer TransformReaderType;
    TransformReaderType reader = itk::TransformFileReader::New();
    reader->SetFileName( filename.c_str() );
    reader->Update();
    const TransformReaderType1::TransformListType * transforms = reader->GetTransformList();
    const TransformReaderType1::TransformPointer tx = *(transforms->begin());
    return tx->GetInputSpaceDimension();
}

template <typename TransformType, class PrecisionType, unsigned int Dimension>
ANTsTransform<TransformType> new_ants_transform( std::string precision, unsigned int dimension, std::string type)
{   
    // assume type == "AffineTransform"
    //if ( type == "AffineTransform" )
    typedef itk::AffineTransform<PrecisionType,Dimension> AffineTransformType;
    typename AffineTransformType::Pointer transformPointer = AffineTransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    return wrap_transform< TransformType >( basePointer);
}


template <typename TransformType, class PrecisionType, unsigned int Dimension>
void wrapNewANTsTransform(py::module & m, std::string const & suffix) {
    m.def(("new_ants_transform" + suffix).c_str(), &new_ants_transform<TransformType, PrecisionType, Dimension>,
         "Create new ANTsTransform", py::return_value_policy::reference_internal);
}

PYBIND11_MODULE(readTransform, m)
{
    wrapNewANTsTransform<itk::Transform<float, 2, 2>, float, 2>(m, "F2");
    wrapNewANTsTransform<itk::Transform<float, 3, 3>, float, 3>(m, "F3");
    wrapNewANTsTransform<itk::Transform<float, 4, 4>, float, 4>(m, "F4");
    wrapNewANTsTransform<itk::Transform<double, 2, 2>, double, 2>(m, "D2");
    wrapNewANTsTransform<itk::Transform<double, 3, 3>, double, 3>(m, "D3");
    wrapNewANTsTransform<itk::Transform<double, 4, 4>, double, 4>(m, "D4");

    m.def("getTransformFileDimension", &getTransformFileDimension);
}