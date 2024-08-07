
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

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

#include "antsTransform.h"
#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename TransformType, typename VectorImageType, typename PrecisionType, unsigned int Dimension>
AntsTransform<TransformType> antsTransformFromDisplacementField( AntsImage<VectorImageType> & field )
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
  VectorImagePointerType antsrField = field.ptr;

  typename DisplacementFieldType::Pointer itkField = DisplacementFieldType::New();
  itkField->SetRegions( antsrField->GetLargestPossibleRegion() );
  itkField->SetSpacing( antsrField->GetSpacing() );
  itkField->SetOrigin( antsrField->GetOrigin() );
  itkField->SetDirection( antsrField->GetDirection() );
  itkField->AllocateInitialized();

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
  AntsTransform<TransformType> outTransform = { displacementTransform.GetPointer() };
  return outTransform;
}

template <typename TransformType, typename VectorImageType, typename PrecisionType, unsigned int Dimension>
AntsImage<VectorImageType> antsTransformToDisplacementField( AntsTransform<TransformType> & xfrm,
                                                             AntsImage<itk::Image<PrecisionType, Dimension>> & ref )
{
  //typedef itk::Transform<PrecisionType,Dimension,Dimension>                  TransformType;
  using ImageType = typename itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;
  using TransformPointerType = typename TransformType::Pointer;
  using DisplacementFieldTransformType = typename itk::DisplacementFieldTransform<PrecisionType, VectorImageType::ImageDimension>;
  using DisplacementFieldTransformPointerType = typename DisplacementFieldTransformType::Pointer;
  using DisplacementFieldType = typename DisplacementFieldTransformType::DisplacementFieldType;
  using VectorType = typename DisplacementFieldType::PixelType;

  TransformPointerType itkTransform = xfrm.ptr;
  DisplacementFieldTransformPointerType warp = dynamic_cast<DisplacementFieldTransformType *>( itkTransform.GetPointer() ) ;

  ImagePointerType domainImage = ref.ptr;

  typedef typename VectorImageType::Pointer VectorImagePointerType;
  VectorImagePointerType antsrField = VectorImageType::New();
  antsrField->CopyInformation( domainImage );
  antsrField->SetRegions( domainImage->GetLargestPossibleRegion() );
  antsrField->SetNumberOfComponentsPerPixel( Dimension );
  antsrField->AllocateInitialized();

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

  AntsImage<VectorImageType> outImage = { antsrField };
  return outImage;
}

void local_antsTransform(nb::module_ &m) {

    m.def("getTransformParameters", &getTransformParameters<itk::Transform<float, 2, 2>>);
    m.def("getTransformParameters", &getTransformParameters<itk::Transform<float, 3, 3>>);
    m.def("getTransformParameters", &getTransformParameters<itk::Transform<float, 4, 4>>);
    m.def("getTransformParameters", &getTransformParameters<itk::Transform<double,2, 2>>);
    m.def("getTransformParameters", &getTransformParameters<itk::Transform<double,3, 3>>);
    m.def("getTransformParameters", &getTransformParameters<itk::Transform<double,4, 4>>);

    m.def("setTransformParameters", &setTransformParameters<itk::Transform<float, 2, 2>>);
    m.def("setTransformParameters", &setTransformParameters<itk::Transform<float, 3, 3>>);
    m.def("setTransformParameters", &setTransformParameters<itk::Transform<float, 4, 4>>);
    m.def("setTransformParameters", &setTransformParameters<itk::Transform<double,2, 2>>);
    m.def("setTransformParameters", &setTransformParameters<itk::Transform<double,3, 3>>);
    m.def("setTransformParameters", &setTransformParameters<itk::Transform<double,4, 4>>);

    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<float, 2, 2>>);
    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<float, 3, 3>>);
    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<float, 4, 4>>);
    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<double,2, 2>>);
    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<double,3, 3>>);
    m.def("getTransformFixedParameters", &getTransformFixedParameters<itk::Transform<double,4, 4>>);

    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<float, 2, 2>>);
    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<float, 3, 3>>);
    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<float, 4, 4>>);
    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<double,2, 2>>);
    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<double,3, 3>>);
    m.def("setTransformFixedParameters", &setTransformFixedParameters<itk::Transform<double,4, 4>>);


    m.def("transformPoint", &transformPoint<itk::DisplacementFieldTransform<float, 2>>);
    m.def("transformPoint", &transformPoint<itk::DisplacementFieldTransform<float, 3>>);
    m.def("transformPoint", &transformPoint<itk::Transform<float, 2, 2>>);
    m.def("transformPoint", &transformPoint<itk::Transform<float, 3, 3>>);
    m.def("transformPoint", &transformPoint<itk::Transform<float, 4, 4>>);
    m.def("transformPoint", &transformPoint<itk::Transform<double,2, 2>>);
    m.def("transformPoint", &transformPoint<itk::Transform<double,3, 3>>);
    m.def("transformPoint", &transformPoint<itk::Transform<double,4, 4>>);

    m.def("transformVector", &transformVector<itk::Transform<float, 2, 2>>);
    m.def("transformVector", &transformVector<itk::Transform<float, 3, 3>>);
    m.def("transformVector", &transformVector<itk::Transform<float, 4, 4>>);
    m.def("transformVector", &transformVector<itk::Transform<double,2, 2>>);
    m.def("transformVector", &transformVector<itk::Transform<double,3, 3>>);
    m.def("transformVector", &transformVector<itk::Transform<double,4, 4>>);

    m.def("transformImage", &transformImage<itk::Transform<float, 2, 2>, itk::Image<unsigned char, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 3, 3>, itk::Image<unsigned char, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 4, 4>, itk::Image<unsigned char, 4>>);
    m.def("transformImage", &transformImage<itk::Transform<double,2, 2>, itk::Image<unsigned char, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<double,3, 3>, itk::Image<unsigned char, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<double,4, 4>, itk::Image<unsigned char, 4>>);

    m.def("transformImage", &transformImage<itk::Transform<float, 2, 2>, itk::Image<unsigned int, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 3, 3>, itk::Image<unsigned int, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 4, 4>, itk::Image<unsigned int, 4>>);
    m.def("transformImage", &transformImage<itk::Transform<double,2, 2>, itk::Image<unsigned int, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<double,3, 3>, itk::Image<unsigned int, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<double,4, 4>, itk::Image<unsigned int, 4>>);

    m.def("transformImage", &transformImage<itk::Transform<float, 2, 2>, itk::Image<float, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 3, 3>, itk::Image<float, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 4, 4>, itk::Image<float, 4>>);
    m.def("transformImage", &transformImage<itk::Transform<double,2, 2>, itk::Image<float, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<double,3, 3>, itk::Image<float, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<double,4, 4>, itk::Image<float, 4>>);

    m.def("transformImage", &transformImage<itk::Transform<float, 2, 2>, itk::Image<double, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 3, 3>, itk::Image<double, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<float, 4, 4>, itk::Image<double, 4>>);
    m.def("transformImage", &transformImage<itk::Transform<double,2, 2>, itk::Image<double, 2>>);
    m.def("transformImage", &transformImage<itk::Transform<double,3, 3>, itk::Image<double, 3>>);
    m.def("transformImage", &transformImage<itk::Transform<double,4, 4>, itk::Image<double, 4>>);

    // displacement field transforms
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 2>, itk::Image<unsigned char, 2>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 3>, itk::Image<unsigned char, 3>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 2>, itk::Image<unsigned int, 2>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 3>, itk::Image<unsigned int, 3>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 2>, itk::Image<float, 2>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 3>, itk::Image<float, 3>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 2>, itk::Image<double, 2>>);
    m.def("transformImage", &transformImage<itk::DisplacementFieldTransform<float, 3>, itk::Image<double, 3>>);

    m.def("inverseTransform", &inverseTransform<itk::Transform<float, 2, 2>, itk::Transform<float, 2, 2>>);
    m.def("inverseTransform", &inverseTransform<itk::Transform<float, 3, 3>, itk::Transform<float, 3, 3>>);
    m.def("inverseTransform", &inverseTransform<itk::Transform<float, 4, 4>, itk::Transform<float, 4, 4>>);
    m.def("inverseTransform", &inverseTransform<itk::Transform<double,2, 2>, itk::Transform<double,2, 2>>);
    m.def("inverseTransform", &inverseTransform<itk::Transform<double,3, 3>, itk::Transform<double,3, 3>>);
    m.def("inverseTransform", &inverseTransform<itk::Transform<double,4, 4>, itk::Transform<double,4, 4>>);

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

    m.def("writeTransform", &writeTransform<itk::Transform<float, 2, 2>>);
    m.def("writeTransform", &writeTransform<itk::Transform<float, 3, 3>>);
    m.def("writeTransform", &writeTransform<itk::Transform<float, 4, 4>>);
    m.def("writeTransform", &writeTransform<itk::Transform<double,2, 2>>);
    m.def("writeTransform", &writeTransform<itk::Transform<double,3, 3>>);
    m.def("writeTransform", &writeTransform<itk::Transform<double,4, 4>>);

    m.def("matrixOffsetF2", &matrixOffset<itk::Transform<float, 2, 2>, float, 2>);
    m.def("matrixOffsetF3", &matrixOffset<itk::Transform<float, 3, 3>, float, 3>);
    m.def("matrixOffsetF4", &matrixOffset<itk::Transform<float, 4, 4>, float, 4>);
    m.def("matrixOffsetD2", &matrixOffset<itk::Transform<double,2, 2>, double,2>);
    m.def("matrixOffsetD3", &matrixOffset<itk::Transform<double,3, 3>, double,3>);
    m.def("matrixOffsetD4", &matrixOffset<itk::Transform<double,4, 4>, double,4>);

    m.def("antsTransformFromDisplacementField", &antsTransformFromDisplacementField<itk::DisplacementFieldTransform<float,2>, itk::VectorImage<float,2>,float,2>);
    m.def("antsTransformFromDisplacementField", &antsTransformFromDisplacementField<itk::DisplacementFieldTransform<float,3>, itk::VectorImage<float,3>,float,3>);
    m.def("antsTransformToDisplacementField", &antsTransformToDisplacementField<itk::DisplacementFieldTransform<float,2>, itk::VectorImage<float,2>,float,2>);
    m.def("antsTransformToDisplacementField", &antsTransformToDisplacementField<itk::DisplacementFieldTransform<float,3>, itk::VectorImage<float,3>,float,3>);


    nb::class_<AntsTransform<itk::DisplacementFieldTransform<float,2>>>(m, "AntsTransformDF2");
    nb::class_<AntsTransform<itk::DisplacementFieldTransform<float,3>>>(m, "AntsTransformDF3");
    nb::class_<AntsTransform<itk::Transform<float, 2, 2>>>(m, "AntsTransformF22");
    nb::class_<AntsTransform<itk::Transform<float, 3, 3>>>(m, "AntsTransformF33");
    nb::class_<AntsTransform<itk::Transform<float, 4, 4>>>(m, "AntsTransformF44");
    nb::class_<AntsTransform<itk::Transform<double,2, 2>>>(m, "AntsTransformD22");
    nb::class_<AntsTransform<itk::Transform<double,3, 3>>>(m, "AntsTransformD33");
    nb::class_<AntsTransform<itk::Transform<double,4, 4>>>(m, "AntsTransformD44");

}



