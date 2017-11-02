
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkVectorCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkVectorToRGBImageAdaptor.h"
#include "itkRGBToVectorImageAdaptor.h"
#include "itkCastImageFilter.h"

#include "LOCAL_mergeChannels.cxx"
#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <typename VectorImageType, typename ImageOfVectorsType, typename PixelType, unsigned int Dimension>
typename VectorImageType::Pointer imageOfVectors_to_VectorImage( typename ImageOfVectorsType::Pointer imageOfVectors )
{
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  typedef typename ImageOfVectorsType::Pointer ImageOfVectorsPointerType;

  typedef itk::Image<PixelType, Dimension> ImageType;

  // split channels
  std::vector<py::capsule> myChannels;
  py::capsule myCapsule = wrap<ImageOfVectorsType>(imageOfVectors);
  myChannels = splitChannels<ImageOfVectorsType,ImageType>( myCapsule );

  // now merge channels into a vectorimage
  py::capsule finalImage = mergeChannels2<ImageType, VectorImageType>( myChannels );
  return as<VectorImageType>( finalImage );
}

template <typename VectorImageType, typename ImageOfVectorsType, typename PixelType, unsigned int Dimension>
typename ImageOfVectorsType::Pointer VectorImage_to_imageOfVectors( typename VectorImageType::Pointer vectorImage )
{
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  typedef typename ImageOfVectorsType::Pointer ImageOfVectorsPointerType;

  typedef itk::Image<PixelType, Dimension> ImageType;

  // split channels
  std::vector<py::capsule> myChannels;
  py::capsule myCapsule = wrap<VectorImageType>(vectorImage);
  myChannels = splitChannels<VectorImageType,ImageType>( myCapsule );

  // now merge channels into a vectorimage
  py::capsule finalImage = mergeChannels3<ImageType, ImageOfVectorsType>( myChannels );
  return as<ImageOfVectorsType>( finalImage );
}


template <typename RGBImageType, unsigned int Dimension>
py::capsule RgbToVector( py::capsule antsImage )
{

  typedef typename RGBImageType::Pointer RGBImagePointerType;

  typedef itk::Image<itk::Vector<unsigned char,3>, Dimension>  VectorImageType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  RGBImagePointerType rgbImage = as< RGBImageType >( antsImage );
  
  // 2) Cast to Vector image for processing
  typedef itk::RGBToVectorImageAdaptor<RGBImageType> AdaptorInputType;
  typename AdaptorInputType::Pointer adaptInput = AdaptorInputType::New();
  adaptInput->SetImage( rgbImage );

  typedef itk::CastImageFilter<AdaptorInputType,VectorImageType> CastInputType;
  typename CastInputType::Pointer castInput = CastInputType::New();
  castInput->SetInput( adaptInput );
  castInput->Update();

  // get an image of vectors from rgb
  VectorImagePointerType vectorImage = castInput->GetOutput();

  // cast image of vectors to vectorimage
  typedef itk::VectorImage<unsigned char, Dimension>  VectorImageType2;
  typedef typename VectorImageType2::Pointer VectorImagePointerType2;

  VectorImagePointerType2 newVectorImage = imageOfVectors_to_VectorImage<VectorImageType2,VectorImageType,unsigned char, Dimension>(vectorImage);
  return wrap<VectorImageType2>( newVectorImage );
}

template <typename VectorImageType, unsigned int Dimension>
py::capsule VectorToRgb( py::capsule antsImage )
{
  // cast VectorImage to ImageOfVectors
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  VectorImagePointerType vectorImage = as<VectorImageType>( antsImage );

  typedef itk::Image<itk::Vector<unsigned char,3>, Dimension>  VectorImageType2;
  typedef typename VectorImageType2::Pointer VectorImagePointerType2;

  VectorImagePointerType2 newVectorImage = VectorImage_to_imageOfVectors<VectorImageType,VectorImageType2,unsigned char, Dimension>(vectorImage);
  
  typedef itk::Image< itk::RGBPixel<unsigned char>, Dimension > RGBImageType;
  typedef typename RGBImageType::Pointer RGBImagePointerType;
  
  // 2) Cast to Vector image for processing
  typedef itk::VectorToRGBImageAdaptor<VectorImageType2> AdaptorInputType;
  typename AdaptorInputType::Pointer adaptInput = AdaptorInputType::New();
  adaptInput->SetImage( newVectorImage );
 
  typedef itk::CastImageFilter<AdaptorInputType,RGBImageType> CastInputType;
  typename CastInputType::Pointer castInput = CastInputType::New();
  castInput->SetInput( adaptInput );
  castInput->Update();
  
  return wrap<RGBImageType>( castInput->GetOutput() );

}

PYBIND11_MODULE(rgbToVector, m)
{
  m.def("RgbToVector2", &RgbToVector<itk::Image<itk::RGBPixel<unsigned char>, 2>, 2>);
  m.def("RgbToVector3", &RgbToVector<itk::Image<itk::RGBPixel<unsigned char>, 3>, 3>);
  m.def("VectorToRgb2", &VectorToRgb<itk::VectorImage<unsigned char,2>, 2>);
  m.def("VectorToRgb3", &VectorToRgb<itk::VectorImage<unsigned char,3>, 3>);
}



