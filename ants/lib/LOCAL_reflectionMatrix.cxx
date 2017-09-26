
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkAffineTransform.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageMomentsCalculator.h"
#include "itkTransformFileWriter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template< class ImageType >
int reflectionMatrixHelper( py::capsule py_image, unsigned int axis, std::string filename )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef itk::AffineTransform<double, ImageType::ImageDimension> AffineTransformType;

  ImagePointerType image = as<ImageType>( py_image );

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
int reflectionMatrixEntry( py::capsule image, unsigned int axis, std::string filename)
{
  return reflectionMatrixHelper<ImageType>( image, axis, filename );
}



PYBIND11_MODULE(reflectionMatrix, m)
{
  m.def("reflectionMatrixUC2", &reflectionMatrixEntry<itk::Image<unsigned char, 2>>);
  m.def("reflectionMatrixUC3", &reflectionMatrixEntry<itk::Image<unsigned char, 3>>);
  m.def("reflectionMatrixUC4", &reflectionMatrixEntry<itk::Image<unsigned char, 4>>);
  m.def("reflectionMatrixUI2", &reflectionMatrixEntry<itk::Image<unsigned int, 2>>);
  m.def("reflectionMatrixUI3", &reflectionMatrixEntry<itk::Image<unsigned int, 3>>);
  m.def("reflectionMatrixUI4", &reflectionMatrixEntry<itk::Image<unsigned int, 4>>);
  m.def("reflectionMatrixF2", &reflectionMatrixEntry<itk::Image<float, 2>>);
  m.def("reflectionMatrixF3", &reflectionMatrixEntry<itk::Image<float, 3>>);
  m.def("reflectionMatrixF4", &reflectionMatrixEntry<itk::Image<float, 4>>);
  m.def("reflectionMatrixD2", &reflectionMatrixEntry<itk::Image<double, 2>>);
  m.def("reflectionMatrixD3", &reflectionMatrixEntry<itk::Image<double, 3>>);
  m.def("reflectionMatrixD4", &reflectionMatrixEntry<itk::Image<double, 4>>);
}

