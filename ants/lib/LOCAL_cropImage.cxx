
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdio.h>
#include "itkCastImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLabelStatisticsImageFilter.h"
#include "itkPasteImageFilter.h"
#include <string>
#include <vector>

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template< class ImageType >
typename ImageType::Pointer cropImageHelper(  typename ImageType::Pointer image,
                                              typename ImageType::Pointer labimage,
                                              unsigned int whichLabel  )
{
  enum { Dimension = ImageType::ImageDimension };
  typename ImageType::RegionType region;
  if( image.IsNotNull() & labimage.IsNotNull() )
    {
    typedef itk::Image<unsigned short, Dimension>      ShortImageType;
    typedef itk::CastImageFilter<ImageType, ShortImageType> CasterType;
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput( labimage );
    caster->Update();
    typedef itk::LabelStatisticsImageFilter<ShortImageType, ShortImageType>
      StatsFilterType;
    typename StatsFilterType::Pointer stats = StatsFilterType::New();
    stats->SetLabelInput( caster->GetOutput() );
    stats->SetInput( caster->GetOutput() );
    stats->Update();
    region = stats->GetRegion( whichLabel );
    typedef itk::ExtractImageFilter<ImageType, ImageType> CropperType;
    typename CropperType::Pointer cropper = CropperType::New();
    cropper->SetInput( image );
    cropper->SetExtractionRegion( region );
    cropper->SetDirectionCollapseToSubmatrix();
    cropper->UpdateLargestPossibleRegion();
    cropper->GetOutput()->SetSpacing( image->GetSpacing() );
    typename ImageType::RegionType region =
      cropper->GetOutput()->GetLargestPossibleRegion();
    typename ImageType::IndexType ind = region.GetIndex();
    typename ImageType::PointType neworig;
    image->TransformIndexToPhysicalPoint( ind, neworig );
    ind.Fill(0);
    region.SetIndex( ind );
    cropper->GetOutput()->SetRegions( region );
    cropper->GetOutput()->SetOrigin( neworig );
    return cropper->GetOutput();
    }
  return nullptr;
}


template< class ImageType >
typename ImageType::Pointer cropIndHelper(  typename ImageType::Pointer image,
                                            std::vector<int> lindv, std::vector<int> uindv )
{
  enum { Dimension = ImageType::ImageDimension };

  typename ImageType::RegionType region;
  typename ImageType::RegionType::SizeType size;
  typename ImageType::IndexType loind;
  typename ImageType::IndexType upind;
  typename ImageType::IndexType index;
  for( int i = 0 ; i < Dimension; ++i )
    {
    loind[i] = lindv[i];// - 1;
    upind[i] = uindv[i];// - 1; // R uses a different indexing, by 1 instead of 0
    if ( upind[i] > loind[i] )
      {
      size[i] = upind[i] - loind[i];// + 1;
      index[i] = loind[i];
      }
    else
      {
      size[i] = loind[i] - upind[i];// + 1;
      index[i] = upind[i];
      }
    }
  if( image.IsNotNull() )
    {
    region.SetSize( size );
    region.SetIndex( index );
    typedef itk::ExtractImageFilter<ImageType, ImageType> CropperType;
    typename CropperType::Pointer cropper = CropperType::New();
    cropper->SetInput( image );
    cropper->SetExtractionRegion( region );
    cropper->SetDirectionCollapseToSubmatrix();
    cropper->Update();
    cropper->GetOutput()->SetSpacing( image->GetSpacing() );
    typename ImageType::RegionType region =
      cropper->GetOutput()->GetLargestPossibleRegion();
    typename ImageType::IndexType ind = region.GetIndex();
    typename ImageType::PointType neworig;
    image->TransformIndexToPhysicalPoint( ind, neworig );
    ind.Fill(0);
    region.SetIndex( ind );
    cropper->GetOutput()->SetRegions( region );
    cropper->GetOutput()->SetOrigin( neworig );
    return cropper->GetOutput();
    }
  return nullptr;
}

template< class ImageType >
typename ImageType::Pointer decropImageHelper(  typename ImageType::Pointer cimage,
                                                typename ImageType::Pointer fimage )
{
  enum { Dimension = ImageType::ImageDimension };
  typename ImageType::RegionType region;
  if( cimage.IsNotNull() & fimage.IsNotNull() )
    {
    typedef itk::PasteImageFilter <ImageType, ImageType >
      PasteImageFilterType;
    // The SetDestinationIndex() method prescribes where in the first
    // input to start pasting data from the second input.
    // The SetSourceRegion method prescribes the section of the second
    // image to paste into the first.
    typename ImageType::IndexType destinationIndex;
    fimage->TransformPhysicalPointToIndex(
      cimage->GetOrigin(), destinationIndex );
//      cimage->GetLargestPossibleRegion().GetIndex();
    typename PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();
    pasteFilter->SetSourceImage(cimage);
    pasteFilter->SetDestinationImage(fimage);
    pasteFilter->SetSourceRegion(cimage->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->Update();
    return pasteFilter->GetOutput();
    }
  return nullptr;
}

template <typename ImageType>
py::capsule cropImage( py::capsule &in_image1,
                                  py::capsule &in_image2,
                                  unsigned int label,
                                  unsigned int decrop,
                                  std::vector<int> loind,
                                  std::vector<int> upind  )
{
  typedef typename ImageType::Pointer ImagePointerType;

  ImagePointerType antsimage1 = as< ImageType >( in_image1 );
  ImagePointerType antsimage2 = as< ImageType >( in_image2 );

  if ( decrop == 0 )
  {
    ImagePointerType out_image = cropImageHelper<ImageType>(antsimage1, antsimage2, label);
    py::capsule out_ants_image = wrap<ImageType>( out_image );
    return out_ants_image;
  }
  else if ( decrop == 1 )
  {
    ImagePointerType out_image = decropImageHelper<ImageType>(antsimage1, antsimage2);
    py::capsule out_ants_image = wrap<ImageType>( out_image );
    return out_ants_image;
  }

  else if ( decrop == 2 )
  {
    ImagePointerType out_image = cropIndHelper<ImageType>(antsimage1, loind, upind);
    py::capsule out_ants_image = wrap<ImageType>( out_image );
    return out_ants_image;
  }

  ImagePointerType out_image = ImageType::New();
  py::capsule out_ants_image = wrap<ImageType>( out_image );
  return out_ants_image;

}



PYBIND11_MODULE(cropImage, m) {
  m.def("cropImageF2", &cropImage<itk::Image<float, 2>>);
  m.def("cropImageF3", &cropImage<itk::Image<float, 3>>);
}
