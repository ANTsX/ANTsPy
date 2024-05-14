
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "antsImage.h"

namespace nb = nanobind;

using namespace nb::literals;

template< class ImageType, class VectorImageType >
AntsImage<VectorImageType> mergeChannels( std::vector<AntsImage<ImageType>> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( imageList[i].ptr );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->AllocateInitialized();

  // Fill image data
  itk::ImageRegionIteratorWithIndex<VectorImageType> it( vectorImage,
    vectorImage->GetLargestPossibleRegion() );

  while (!it.IsAtEnd() )
    {
    typename VectorImageType::PixelType pix;
    pix.SetSize( nImages );
    for (unsigned int i=0; i<nImages; i++)
      {
      pix[i] = images[i]->GetPixel(it.GetIndex());
      }
    vectorImage->SetPixel(it.GetIndex(), pix);
    ++it;
    }

    AntsImage<VectorImageType> outImage = { vectorImage };
  return outImage;
}


template< class ImageType, class VectorImageType >
typename VectorImageType::Pointer mergeChannels2( std::vector<AntsImage<ImageType>> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( imageList[i].ptr );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->AllocateInitialized();

  // Fill image data
  itk::ImageRegionIteratorWithIndex<VectorImageType> it( vectorImage,
    vectorImage->GetLargestPossibleRegion() );

  while (!it.IsAtEnd() )
    {
    typename VectorImageType::PixelType pix;
    pix.SetSize( nImages );
    for (unsigned int i=0; i<nImages; i++)
      {
      pix[i] = images[i]->GetPixel(it.GetIndex());
      }
    vectorImage->SetPixel(it.GetIndex(), pix);
    ++it;
    }

  //return wrap<VectorImageType>( vectorImage );
  //AntsImage<VectorImageType> outImage = { vectorImage };
  return vectorImage;
}

template< class ImageType, class VectorImageType >
typename VectorImageType::Pointer mergeChannels3( std::vector<AntsImage<ImageType>> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( imageList[i].ptr );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  //vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->AllocateInitialized();

  // Fill image data
  itk::ImageRegionIteratorWithIndex<VectorImageType> it( vectorImage,
    vectorImage->GetLargestPossibleRegion() );

  while (!it.IsAtEnd() )
    {

    typename VectorImageType::PixelType pix;
    //pix.SetSize( nImages );
    for (unsigned int i=0; i<nImages; i++)
      {
      pix[i] = images[i]->GetPixel(it.GetIndex());
      }
    vectorImage->SetPixel(it.GetIndex(), pix);
    ++it;
    }

  //return wrap<VectorImageType>( vectorImage );
  //AntsImage<VectorImageType> outImage = { vectorImage };
  return vectorImage;
}


template< class VectorImageType, class ImageType>
std::vector<AntsImage<ImageType>> splitChannels( AntsImage<VectorImageType> & antsimage )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  VectorImagePointerType input = antsimage.ptr;
  unsigned int nComponents = input->GetNumberOfComponentsPerPixel();

  // Create output images
  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nComponents; i++)
    {
    ImagePointerType image = ImageType::New();
    image->SetRegions( input->GetLargestPossibleRegion() );
    image->SetSpacing( input->GetSpacing() );
    image->SetOrigin( input->GetOrigin() );
    image->SetDirection( input->GetDirection() );
    image->AllocateInitialized();
    images.push_back( image );
    }

  // Fill image data
  itk::ImageRegionIteratorWithIndex<VectorImageType> it( input,
    input->GetLargestPossibleRegion() );

  while (!it.IsAtEnd() )
    {
    typename VectorImageType::PixelType pix = input->GetPixel(it.GetIndex());

    for (unsigned int i=0; i<nComponents; i++)
      {
      images[i]->SetPixel(it.GetIndex(), pix[i]);
      }
    ++it;
    }

  std::vector<AntsImage<ImageType>> outputList( nComponents );
  for (unsigned int i=0; i<nComponents; i++)
    {
        AntsImage<ImageType> tmpImage = { images[i] };
    outputList[i] = tmpImage;
    }

  return( outputList );

}


void local_mergeChannels(nb::module_ &m) {
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned char, 2>, itk::VectorImage<unsigned char, 2> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned char, 3>, itk::VectorImage<unsigned char, 3> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned char, 4>, itk::VectorImage<unsigned char, 4> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned int, 2>, itk::VectorImage<unsigned int, 2> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned int, 3>, itk::VectorImage<unsigned int, 3> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<unsigned int, 4>, itk::VectorImage<unsigned int, 4> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<float, 2>, itk::VectorImage<float, 2> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<float, 3>, itk::VectorImage<float, 3> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<float, 4>, itk::VectorImage<float, 4> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<double, 2>, itk::VectorImage<double, 2> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<double, 3>, itk::VectorImage<double, 3> >);
  m.def("mergeChannels", &mergeChannels<itk::Image<double, 4>, itk::VectorImage<double, 4> >);

  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned char, 2>, itk::Image<unsigned char, 2> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned char, 3>, itk::Image<unsigned char, 3> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned char, 4>, itk::Image<unsigned char, 4> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned int, 2>, itk::Image<unsigned int, 2> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned int, 3>, itk::Image<unsigned int, 3> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<unsigned int, 4>, itk::Image<unsigned int, 4> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<float, 2>, itk::Image<float, 2> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<float, 3>, itk::Image<float, 3> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<float, 4>, itk::Image<float, 4> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<double, 2>, itk::Image<double, 2> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<double, 3>, itk::Image<double, 3> >);
  m.def("splitChannels", &splitChannels<itk::VectorImage<double, 4>, itk::Image<double, 4> >);
  m.def("splitChannels", &splitChannels<itk::Image<itk::RGBPixel<unsigned char>, 2>, itk::Image<unsigned char, 2> >);


}
