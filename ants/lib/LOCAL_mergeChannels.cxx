
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template< class ImageType, class VectorImageType >
py::capsule mergeChannels( std::vector<void *> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( as<ImageType>( imageList[i] ) );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->Allocate();

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

  return wrap<VectorImageType>( vectorImage );
}

template< class ImageType, class VectorImageType >
py::capsule mergeChannels2( std::vector<py::capsule> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( as<ImageType>( imageList[i] ) );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->Allocate();

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

  return wrap<VectorImageType>( vectorImage );
}

template< class ImageType, class VectorImageType >
py::capsule mergeChannels3( std::vector<py::capsule> imageList )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  unsigned int nImages = imageList.size();

  std::vector<ImagePointerType> images;
  for ( unsigned int i=0; i<nImages; i++)
  {
    images.push_back( as<ImageType>( imageList[i] ) );
  }

  VectorImagePointerType vectorImage = VectorImageType::New();
  vectorImage->SetRegions( images[0]->GetLargestPossibleRegion() );
  vectorImage->SetSpacing( images[0]->GetSpacing() );
  vectorImage->SetOrigin( images[0]->GetOrigin() );
  vectorImage->SetDirection( images[0]->GetDirection() );
  //vectorImage->SetNumberOfComponentsPerPixel( nImages );
  vectorImage->Allocate();

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

  return wrap<VectorImageType>( vectorImage );
}

template< class VectorImageType, class ImageType>
std::vector<py::capsule > splitChannels( py::capsule & antsimage )
{
  typedef typename ImageType::Pointer       ImagePointerType;
  typedef typename VectorImageType::Pointer VectorImagePointerType;

  VectorImagePointerType input = as<VectorImageType>( antsimage );
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
    image->Allocate();
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

  std::vector<py::capsule > outputList( nComponents );
  for (unsigned int i=0; i<nComponents; i++)
    {
    outputList[i] = wrap<ImageType>( images[i] );
    }

  return( outputList );

}


PYBIND11_MODULE(mergeChannels, m)
{
  m.def("mergeChannelsUC2", &mergeChannels<itk::Image<unsigned char, 2>, itk::VectorImage<unsigned char, 2> >);
  m.def("mergeChannelsUC3", &mergeChannels<itk::Image<unsigned char, 3>, itk::VectorImage<unsigned char, 3> >);
  m.def("mergeChannelsUC4", &mergeChannels<itk::Image<unsigned char, 4>, itk::VectorImage<unsigned char, 4> >);
  m.def("mergeChannelsUI2", &mergeChannels<itk::Image<unsigned int, 2>, itk::VectorImage<unsigned int, 2> >);
  m.def("mergeChannelsUI3", &mergeChannels<itk::Image<unsigned int, 3>, itk::VectorImage<unsigned int, 3> >);
  m.def("mergeChannelsUI4", &mergeChannels<itk::Image<unsigned int, 4>, itk::VectorImage<unsigned int, 4> >);
  m.def("mergeChannelsF2", &mergeChannels<itk::Image<float, 2>, itk::VectorImage<float, 2> >);
  m.def("mergeChannelsF3", &mergeChannels<itk::Image<float, 3>, itk::VectorImage<float, 3> >);
  m.def("mergeChannelsF4", &mergeChannels<itk::Image<float, 4>, itk::VectorImage<float, 4> >);

  m.def("splitChannelsVUC2", &splitChannels<itk::VectorImage<unsigned char, 2>, itk::Image<unsigned char, 2> >);
  m.def("splitChannelsVUC3", &splitChannels<itk::VectorImage<unsigned char, 3>, itk::Image<unsigned char, 3> >);
  m.def("splitChannelsVUC4", &splitChannels<itk::VectorImage<unsigned char, 4>, itk::Image<unsigned char, 4> >);
  m.def("splitChannelsVUI2", &splitChannels<itk::VectorImage<unsigned int, 2>, itk::Image<unsigned int, 2> >);
  m.def("splitChannelsVUI3", &splitChannels<itk::VectorImage<unsigned int, 3>, itk::Image<unsigned int, 3> >);
  m.def("splitChannelsVUI4", &splitChannels<itk::VectorImage<unsigned int, 4>, itk::Image<unsigned int, 4> >);
  m.def("splitChannelsVF2", &splitChannels<itk::VectorImage<float, 2>, itk::Image<float, 2> >);
  m.def("splitChannelsVF3", &splitChannels<itk::VectorImage<float, 3>, itk::Image<float, 3> >);
  m.def("splitChannelsVF4", &splitChannels<itk::VectorImage<float, 4>, itk::Image<float, 4> >);
  m.def("splitChannelsRGBUC2", &splitChannels<  itk::Image<itk::RGBPixel<unsigned char>, 2>, itk::Image<unsigned char, 2> >);


}
