
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkAffineTransform.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageMomentsCalculator.h"
#include "itkTransformFileWriter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template< class ImageType >
int reflectionMatrixHelper( AntsImage<ImageType> & py_image, unsigned int axis, std::string filename )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef itk::AffineTransform<double, ImageType::ImageDimension> AffineTransformType;

  ImagePointerType image = py_image.ptr;

  typedef typename itk::ImageMomentsCalculator<ImageType> ImageCalculatorType;
  typename ImageCalculatorType::Pointer calculator = ImageCalculatorType::New();
  calculator->SetImage( image );

  typename ImageCalculatorType::VectorType fixed_center;
  fixed_center.Fill(0);

  calculator->Compute();
  fixed_center = calculator->GetCenterOfGravity();

  typename AffineTransformType::Pointer aff = AffineTransformType::New();
  aff->SetIdentity();

  typename AffineTransformType::ParametersType myoff = aff->GetFixedParameters();
  for( unsigned int i = 0; i < ImageType::ImageDimension; i++ )
    {
    myoff[i] = fixed_center[i];
    }
  typename AffineTransformType::MatrixType mymat = aff->GetMatrix();
  if( axis < ImageType::ImageDimension )
    {
    mymat[axis][axis] = ( -1.0 );
    }

  aff->SetFixedParameters( myoff );
  aff->SetMatrix( mymat );

  typedef itk::TransformFileWriter TransformWriterType;
  typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
  transformWriter->SetInput( aff );
  transformWriter->SetFileName( filename.c_str() );
  transformWriter->Update();

  return 1;
}


template <typename ImageType>
int reflectionMatrixEntry( AntsImage<ImageType> & image, unsigned int axis, std::string filename)
{
  return reflectionMatrixHelper<ImageType>( image, axis, filename );
}



void local_reflectionMatrix(nb::module_ &m)
{
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned char, 2>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned char, 3>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned char, 4>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned int, 2>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned int, 3>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<unsigned int, 4>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<float, 2>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<float, 3>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<float, 4>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<double, 2>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<double, 3>>);
  m.def("reflectionMatrix", &reflectionMatrixEntry<itk::Image<double, 4>>);
}

