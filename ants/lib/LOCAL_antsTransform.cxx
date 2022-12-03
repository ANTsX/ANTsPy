
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <vector>
#include <string>

#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkVector.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "vnl/vnl_vector_ref.h"
#include "itkTransform.h"
#include "itkAffineTransform.h"

#include "itkAffineTransform.h"
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

#include "antscore/antsUtilities.h"

#include "LOCAL_antsTransform.h"
#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <typename TransformType, typename VectorImageType, typename PrecisionType, unsigned int Dimension>
py::capsule antsTransformFromDisplacementField( py::capsule field )
{
  //typedef itk::Transform<PrecisionType,Dimension,Dimension>                  TransformType;
  typedef typename TransformType::Pointer                                    TransformPointerType;
  typedef typename itk::DisplacementFieldTransform<PrecisionType, Dimension> DisplacementFieldTransformType;
  typedef typename DisplacementFieldTransformType::DisplacementFieldType     DisplacementFieldType;
  typedef typename DisplacementFieldType::PixelType                          VectorType;

  // Displacement field is an itk::Image with vector pixels, while in ANTsR we use the
  // itk::VectorImage class for multichannel data. So we must copy the field
  // and pass it to the transform
  //typedef itk::VectorImage<PrecisionType, Dimension> AntsrFieldType;
  //typedef typename AntsrFieldType::Pointer           AntsrFieldPointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  VectorImagePointerType antsrField = as< VectorImageType >( field );

  typename DisplacementFieldType::Pointer itkField = DisplacementFieldType::New();
  itkField->SetRegions( antsrField->GetLargestPossibleRegion() );
  itkField->SetSpacing( antsrField->GetSpacing() );
  itkField->SetOrigin( antsrField->GetOrigin() );
  itkField->SetDirection( antsrField->GetDirection() );
  itkField->Allocate();

  typedef itk::ImageRegionIteratorWithIndex<DisplacementFieldType> IteratorType;
  IteratorType it( itkField, itkField->GetLargestPossibleRegion() );
  while ( !it.IsAtEnd() )
  {
    typename VectorImageType::PixelType vec = antsrField->GetPixel( it.GetIndex() );
    VectorType dvec;
    for ( unsigned int i=0; i<Dimension; i++)
      {
      dvec[i] = vec[i];
      }
    itkField->SetPixel(it.GetIndex(), dvec);
    ++it;
  }

  typename DisplacementFieldTransformType::Pointer displacementTransform =
    DisplacementFieldTransformType::New();
  displacementTransform->SetDisplacementField( itkField );

  /*
  TransformPointerType transform = dynamic_cast<TransformType *>( displacementTransform.GetPointer() );

  Rcpp::S4 antsrTransform( "antsrTransform" );
  antsrTransform.slot("dimension") = Dimension;
  antsrTransform.slot("precision") = precision;
  std::string type = displacementTransform->GetNameOfClass();
  antsrTransform.slot("type") = type;
  TransformPointerType * rawPointer = new TransformPointerType( transform );
  Rcpp::XPtr<TransformPointerType> xptr( rawPointer, true );
  antsrTransform.slot("pointer") = xptr;

  return antsrTransform;
  */
  return wrap_transform< TransformType >( displacementTransform.GetPointer() );
}

template <typename TransformType, typename VectorImageType, typename PrecisionType, unsigned int Dimension>
py::capsule antsTransformToDisplacementField( py::capsule xfrm, py::capsule ref )
{
  //typedef itk::Transform<PrecisionType,Dimension,Dimension>                  TransformType;
  using ImageType = typename itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;
  using TransformPointerType = typename TransformType::Pointer;
  using DisplacementFieldTransformType = typename itk::DisplacementFieldTransform<PrecisionType, VectorImageType::ImageDimension>;
  using DisplacementFieldTransformPointerType = typename DisplacementFieldTransformType::Pointer;
  using DisplacementFieldType = typename DisplacementFieldTransformType::DisplacementFieldType;
  using VectorType = typename DisplacementFieldType::PixelType;

  TransformPointerType itkTransform = as_transform<TransformType>( xfrm );
  DisplacementFieldTransformPointerType warp = dynamic_cast<DisplacementFieldTransformType *>( itkTransform.GetPointer() ) ;

  ImagePointerType domainImage = as<ImageType>( ref );

  typedef typename VectorImageType::Pointer VectorImagePointerType;
  VectorImagePointerType antsrField = VectorImageType::New();
  antsrField->CopyInformation( domainImage );
  antsrField->SetRegions( domainImage->GetLargestPossibleRegion() );
  antsrField->SetNumberOfComponentsPerPixel( Dimension );
  antsrField->Allocate();

  typedef itk::ImageRegionIteratorWithIndex<ImageType> IteratorType;
  IteratorType it( domainImage, domainImage->GetLargestPossibleRegion() );
  while ( !it.IsAtEnd() )
    {
    VectorType vec = warp->GetDisplacementField()->GetPixel( it.GetIndex() );
    typename VectorImageType::PixelType dvec;
    dvec.SetSize( Dimension );
    for( unsigned int i = 0; i < Dimension; i++ )
      {
      dvec[i] = vec[i];
      }
    antsrField->SetPixel( it.GetIndex(), dvec );
    ++it;
    }

  return wrap< VectorImageType >( antsrField );
}

PYBIND11_MODULE(antsTransform, m) {

    m.def("getTransformParametersF2", &getTransformParameters<itk::Transform<float, 2, 2>>);
    m.def("getTransformParametersF3", &getTransformParameters<itk::Transform<float, 3, 3>>);
    m.def("getTransformParametersF4", &getTransformParameters<itk::Transform<float, 4, 4>>);
    m.def("getTransformParametersD2", &getTransformParameters<itk::Transform<double,2, 2>>);
    m.def("getTransformParametersD3", &getTransformParameters<itk::Transform<double,3, 3>>);
    m.def("getTransformParametersD4", &getTransformParameters<itk::Transform<double,4, 4>>);

    m.def("setTransformParametersF2", &setTransformParameters<itk::Transform<float, 2, 2>>);
    m.def("setTransformParametersF3", &setTransformParameters<itk::Transform<float, 3, 3>>);
    m.def("setTransformParametersF4", &setTransformParameters<itk::Transform<float, 4, 4>>);
    m.def("setTransformParametersD2", &setTransformParameters<itk::Transform<double,2, 2>>);
    m.def("setTransformParametersD3", &setTransformParameters<itk::Transform<double,3, 3>>);
    m.def("setTransformParametersD4", &setTransformParameters<itk::Transform<double,4, 4>>);

    m.def("getTransformFixedParametersF2", &getTransformFixedParameters<itk::Transform<float, 2, 2>>);
    m.def("getTransformFixedParametersF3", &getTransformFixedParameters<itk::Transform<float, 3, 3>>);
    m.def("getTransformFixedParametersF4", &getTransformFixedParameters<itk::Transform<float, 4, 4>>);
    m.def("getTransformFixedParametersD2", &getTransformFixedParameters<itk::Transform<double,2, 2>>);
    m.def("getTransformFixedParametersD3", &getTransformFixedParameters<itk::Transform<double,3, 3>>);
    m.def("getTransformFixedParametersD4", &getTransformFixedParameters<itk::Transform<double,4, 4>>);

    m.def("setTransformFixedParametersF2", &setTransformFixedParameters<itk::Transform<float, 2, 2>>);
    m.def("setTransformFixedParametersF3", &setTransformFixedParameters<itk::Transform<float, 3, 3>>);
    m.def("setTransformFixedParametersF4", &setTransformFixedParameters<itk::Transform<float, 4, 4>>);
    m.def("setTransformFixedParametersD2", &setTransformFixedParameters<itk::Transform<double,2, 2>>);
    m.def("setTransformFixedParametersD3", &setTransformFixedParameters<itk::Transform<double,3, 3>>);
    m.def("setTransformFixedParametersD4", &setTransformFixedParameters<itk::Transform<double,4, 4>>);

    m.def("transformPointF2", &transformPoint<itk::Transform<float, 2, 2>>);
    m.def("transformPointF3", &transformPoint<itk::Transform<float, 3, 3>>);
    m.def("transformPointF4", &transformPoint<itk::Transform<float, 4, 4>>);
    m.def("transformPointD2", &transformPoint<itk::Transform<double,2, 2>>);
    m.def("transformPointD3", &transformPoint<itk::Transform<double,3, 3>>);
    m.def("transformPointD4", &transformPoint<itk::Transform<double,4, 4>>);

    m.def("transformVectorF2", &transformVector<itk::Transform<float, 2, 2>>);
    m.def("transformVectorF3", &transformVector<itk::Transform<float, 3, 3>>);
    m.def("transformVectorF4", &transformVector<itk::Transform<float, 4, 4>>);
    m.def("transformVectorD2", &transformVector<itk::Transform<double,2, 2>>);
    m.def("transformVectorD3", &transformVector<itk::Transform<double,3, 3>>);
    m.def("transformVectorD4", &transformVector<itk::Transform<double,4, 4>>);

    m.def("transformImageF2UC2", &transformImage<itk::Transform<float, 2, 2>, itk::Image<unsigned char, 2>>);
    m.def("transformImageF3UC3", &transformImage<itk::Transform<float, 3, 3>, itk::Image<unsigned char, 3>>);
    m.def("transformImageF4UC4", &transformImage<itk::Transform<float, 4, 4>, itk::Image<unsigned char, 4>>);
    m.def("transformImageD2UC4", &transformImage<itk::Transform<double,2, 2>, itk::Image<unsigned char, 2>>);
    m.def("transformImageD3UC4", &transformImage<itk::Transform<double,3, 3>, itk::Image<unsigned char, 3>>);
    m.def("transformImageD4UC4", &transformImage<itk::Transform<double,4, 4>, itk::Image<unsigned char, 4>>);
    
    m.def("transformImageF2UI2", &transformImage<itk::Transform<float, 2, 2>, itk::Image<unsigned int, 2>>);
    m.def("transformImageF3UI3", &transformImage<itk::Transform<float, 3, 3>, itk::Image<unsigned int, 3>>);
    m.def("transformImageF4UI4", &transformImage<itk::Transform<float, 4, 4>, itk::Image<unsigned int, 4>>);
    m.def("transformImageD2UI4", &transformImage<itk::Transform<double,2, 2>, itk::Image<unsigned int, 2>>);
    m.def("transformImageD3UI4", &transformImage<itk::Transform<double,3, 3>, itk::Image<unsigned int, 3>>);
    m.def("transformImageD4UI4", &transformImage<itk::Transform<double,4, 4>, itk::Image<unsigned int, 4>>);

    m.def("transformImageF2F2", &transformImage<itk::Transform<float, 2, 2>, itk::Image<float, 2>>);
    m.def("transformImageF3F3", &transformImage<itk::Transform<float, 3, 3>, itk::Image<float, 3>>);
    m.def("transformImageF4F4", &transformImage<itk::Transform<float, 4, 4>, itk::Image<float, 4>>);
    m.def("transformImageD2F4", &transformImage<itk::Transform<double,2, 2>, itk::Image<float, 2>>);
    m.def("transformImageD3F4", &transformImage<itk::Transform<double,3, 3>, itk::Image<float, 3>>);
    m.def("transformImageD4F4", &transformImage<itk::Transform<double,4, 4>, itk::Image<float, 4>>);

    m.def("transformImageF2D2", &transformImage<itk::Transform<float, 2, 2>, itk::Image<double, 2>>);
    m.def("transformImageF3D3", &transformImage<itk::Transform<float, 3, 3>, itk::Image<double, 3>>);
    m.def("transformImageF4D4", &transformImage<itk::Transform<float, 4, 4>, itk::Image<double, 4>>);
    m.def("transformImageD2D4", &transformImage<itk::Transform<double,2, 2>, itk::Image<double, 2>>);
    m.def("transformImageD3D4", &transformImage<itk::Transform<double,3, 3>, itk::Image<double, 3>>);
    m.def("transformImageD4D4", &transformImage<itk::Transform<double,4, 4>, itk::Image<double, 4>>);

    m.def("inverseTransformF2", &inverseTransform<itk::Transform<float, 2, 2>, itk::Transform<float, 2, 2>>);
    m.def("inverseTransformF3", &inverseTransform<itk::Transform<float, 3, 3>, itk::Transform<float, 3, 3>>);
    m.def("inverseTransformF4", &inverseTransform<itk::Transform<float, 4, 4>, itk::Transform<float, 4, 4>>);
    m.def("inverseTransformD2", &inverseTransform<itk::Transform<double,2, 2>, itk::Transform<double,2, 2>>);
    m.def("inverseTransformD3", &inverseTransform<itk::Transform<double,3, 3>, itk::Transform<double,3, 3>>);
    m.def("inverseTransformD4", &inverseTransform<itk::Transform<double,4, 4>, itk::Transform<double,4, 4>>);

    m.def("composeTransformsF2", &composeTransforms<itk::Transform<float, 2, 2>, float, 2>);
    m.def("composeTransformsF3", &composeTransforms<itk::Transform<float, 3, 3>, float, 3>);
    m.def("composeTransformsF4", &composeTransforms<itk::Transform<float, 4, 4>, float, 4>);
    m.def("composeTransformsD2", &composeTransforms<itk::Transform<double,2, 2>, double,2> );
    m.def("composeTransformsD3", &composeTransforms<itk::Transform<double,3, 3>, double,3> );
    m.def("composeTransformsD4", &composeTransforms<itk::Transform<double,4, 4>, double,4> );

    m.def("readTransformF2", &readTransform<itk::Transform<float, 2, 2>, float, 2>);
    m.def("readTransformF3", &readTransform<itk::Transform<float, 3, 3>, float, 3>);
    m.def("readTransformF4", &readTransform<itk::Transform<float, 4, 4>, float, 4>);
    m.def("readTransformD2", &readTransform<itk::Transform<double,2, 2>, double,2> );
    m.def("readTransformD3", &readTransform<itk::Transform<double,3, 3>, double,3> );
    m.def("readTransformD4", &readTransform<itk::Transform<double,4, 4>, double,4> );

    m.def("writeTransformF2", &writeTransform<itk::Transform<float, 2, 2>>);
    m.def("writeTransformF3", &writeTransform<itk::Transform<float, 3, 3>>);
    m.def("writeTransformF4", &writeTransform<itk::Transform<float, 4, 4>>);
    m.def("writeTransformD2", &writeTransform<itk::Transform<double,2, 2>>);
    m.def("writeTransformD3", &writeTransform<itk::Transform<double,3, 3>>);
    m.def("writeTransformD4", &writeTransform<itk::Transform<double,4, 4>>);

    m.def("matrixOffsetF2", &matrixOffset<itk::Transform<float, 2, 2>, float, 2>);
    m.def("matrixOffsetF3", &matrixOffset<itk::Transform<float, 3, 3>, float, 3>);
    m.def("matrixOffsetF4", &matrixOffset<itk::Transform<float, 4, 4>, float, 4>);
    m.def("matrixOffsetD2", &matrixOffset<itk::Transform<double,2, 2>, double,2>);
    m.def("matrixOffsetD3", &matrixOffset<itk::Transform<double,3, 3>, double,3>);
    m.def("matrixOffsetD4", &matrixOffset<itk::Transform<double,4, 4>, double,4>);

    m.def("antsTransformFromDisplacementFieldF2", &antsTransformFromDisplacementField<itk::DisplacementFieldTransform<float,2>, itk::VectorImage<float,2>,float,2>);
    m.def("antsTransformFromDisplacementFieldF3", &antsTransformFromDisplacementField<itk::DisplacementFieldTransform<float,3>, itk::VectorImage<float,3>,float,3>);
    m.def("antsTransformToDisplacementFieldF2", &antsTransformToDisplacementField<itk::DisplacementFieldTransform<float,2>, itk::VectorImage<float,2>,float,2>);
    m.def("antsTransformToDisplacementFieldF3", &antsTransformToDisplacementField<itk::DisplacementFieldTransform<float,3>, itk::VectorImage<float,3>,float,3>);

}



