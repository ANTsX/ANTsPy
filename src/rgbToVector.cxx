
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkVectorCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkVectorToRGBImageAdaptor.h"
#include "itkRGBToVectorImageAdaptor.h"
#include "itkCastImageFilter.h"

#include "mergeChannels.h"
#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename VectorImageType, typename ImageOfVectorsType, typename PixelType, unsigned int Dimension>
typename VectorImageType::Pointer imageOfVectors_to_VectorImage( typename ImageOfVectorsType::Pointer imageOfVectors )
{
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  typedef typename ImageOfVectorsType::Pointer ImageOfVectorsPointerType;

  typedef itk::Image<PixelType, Dimension> ImageType;

  // split channels
  std::vector<AntsImage<ImageType>> myChannels;
  AntsImage<ImageOfVectorsType> myCapsule = { imageOfVectors };
  myChannels = splitChannels<ImageOfVectorsType,ImageType>( myCapsule );

  // now merge channels into a vectorimage
  VectorImagePointerType finalImage = mergeChannels2<ImageType, VectorImageType>( myChannels );
  return finalImage;
}

template <typename VectorImageType, typename ImageOfVectorsType, typename PixelType, unsigned int Dimension>
typename ImageOfVectorsType::Pointer VectorImage_to_imageOfVectors( typename VectorImageType::Pointer vectorImage )
{
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  typedef typename ImageOfVectorsType::Pointer ImageOfVectorsPointerType;

  typedef itk::Image<PixelType, Dimension> ImageType;

  // split channels
  std::vector<AntsImage<ImageType>> myChannels;
  AntsImage<VectorImageType> myCapsule = { vectorImage };
  myChannels = splitChannels<VectorImageType,ImageType>( myCapsule );

  // now merge channels into a vectorimage
  ImageOfVectorsPointerType finalImage = mergeChannels3<ImageType, ImageOfVectorsType>( myChannels );
  return finalImage;
}


template <typename RGBImageType, unsigned int Dimension>
AntsImage<itk::VectorImage<unsigned char, Dimension>> RgbToVector( AntsImage<RGBImageType> & antsImage )
{

  typedef typename RGBImageType::Pointer RGBImagePointerType;

  typedef itk::Image<itk::Vector<unsigned char,3>, Dimension>  VectorImageType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  RGBImagePointerType rgbImage = antsImage.ptr;//as< RGBImageType >( antsImage );
  
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
  AntsImage<VectorImageType2> outImage = { newVectorImage };
  return outImage;
}

template <typename VectorImageType, unsigned int Dimension>
AntsImage<itk::Image< itk::RGBPixel<unsigned char>, Dimension >> VectorToRgb( AntsImage<VectorImageType> & antsImage )
{
  // cast VectorImage to ImageOfVectors
  typedef typename VectorImageType::Pointer VectorImagePointerType;
  VectorImagePointerType vectorImage = antsImage.ptr;//as<VectorImageType>( antsImage );

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
  
  //return wrap<RGBImageType>( castInput->GetOutput() );
  AntsImage<RGBImageType> outImage = { castInput->GetOutput() };
  return outImage;

}

void local_rgbToVector(nb::module_ &m)
{
  m.def("RgbToVector2", &RgbToVector<itk::Image<itk::RGBPixel<unsigned char>, 2>, 2>);
  m.def("RgbToVector3", &RgbToVector<itk::Image<itk::RGBPixel<unsigned char>, 3>, 3>);
  m.def("VectorToRgb2", &VectorToRgb<itk::VectorImage<unsigned char,2>, 2>);
  m.def("VectorToRgb3", &VectorToRgb<itk::VectorImage<unsigned char,3>, 3>);
}



