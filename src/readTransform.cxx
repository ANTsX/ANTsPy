#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

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

#include "readTransform.h"

#include "register_transforms.h"

namespace nb = nanobind;
using namespace nb::literals;

unsigned int getTransformDimensionFromFile( std::string filename )
{
    register_transforms();
    typedef itk::TransformFileReader TransformReaderType1;
    typedef typename TransformReaderType1::Pointer TransformReaderType;
    TransformReaderType reader = itk::TransformFileReader::New();
    reader->SetFileName( filename.c_str() );
    reader->Update();
    const TransformReaderType1::TransformListType * transforms = reader->GetTransformList();
    const TransformReaderType1::TransformPointer tx = *(transforms->begin());
    return tx->GetInputSpaceDimension();
}

std::string getTransformNameFromFile( std::string filename )
{
    register_transforms();
    typedef itk::TransformFileReader TransformReaderType1;
    typedef typename TransformReaderType1::Pointer TransformReaderType;
    TransformReaderType reader = itk::TransformFileReader::New();
    reader->SetFileName( filename.c_str() );
    reader->Update();
    const TransformReaderType1::TransformListType * transforms = reader->GetTransformList();
    const TransformReaderType1::TransformPointer tx = *(transforms->begin());
    return std::string( tx->GetNameOfClass() );
}


template <typename PrecisionType, unsigned int Dimension>
AntsTransform<itk::Transform<PrecisionType, Dimension, Dimension>> newAntsTransform( std::string precision, unsigned int dimension, std::string type)
{

    //auto transformPointer = TransformType::New();
  // Initialize transform by type
  if ( type == "AffineTransform" )
    {
    typedef itk::AffineTransform<PrecisionType,Dimension> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "CenteredAffineTransform" )
    {
    typedef itk::CenteredAffineTransform<PrecisionType,Dimension> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "Euler3DTransform" )
    {
    typedef itk::Euler3DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;

    }
  else if ( type == "Euler2DTransform" )
    {
    typedef itk::Euler2DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "QuaternionRigidTransform" )
    {
    typedef itk::QuaternionRigidTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "Rigid2DTransform" )
    {
    typedef itk::Rigid2DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "Rigid3DTransform" )
    {
    typedef itk::Rigid3DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "CenteredEuler3DTransform" )
    {
    typedef itk::CenteredEuler3DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "CenteredRigid2DTransform" )
    {
    typedef itk::CenteredRigid2DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "Similarity3DTransform" )
    {
    typedef itk::Similarity3DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "Similarity2DTransform" )
    {
    typedef itk::Similarity2DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }
  else if ( type == "CenteredSimilarity2DTransform" )
    {
    typedef itk::CenteredSimilarity2DTransform<PrecisionType> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
    }

    typedef itk::AffineTransform<PrecisionType,Dimension> TransformType;
    auto transformPointer = TransformType::New();

    typedef itk::Transform<PrecisionType,Dimension,Dimension> TransformBaseType;
    typedef typename TransformBaseType::Pointer               TransformBasePointerType;
    TransformBasePointerType basePointer = dynamic_cast<TransformBaseType *>( transformPointer.GetPointer() );

    AntsTransform<TransformBaseType> outTransform = { basePointer };
    return outTransform;
}


void local_readTransform(nb::module_ &m)
{
    m.def("newAntsTransformF2", &newAntsTransform<float, 2>);
    m.def("newAntsTransformF3", &newAntsTransform<float, 3>);
    m.def("newAntsTransformD2", &newAntsTransform<double,2>);
    m.def("newAntsTransformD3", &newAntsTransform<double,3>);

    m.def("getTransformDimensionFromFile", &getTransformDimensionFromFile);
    m.def("getTransformNameFromFile", &getTransformNameFromFile);
}
