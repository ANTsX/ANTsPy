
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

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

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

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
    FixNonZeroIndex<ImageType>( cropper->GetOutput() );
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
    FixNonZeroIndex<ImageType>( cropper->GetOutput() );
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
AntsImage<ImageType> cropImage( AntsImage<ImageType> &in_image1,
                                  AntsImage<ImageType> &in_image2,
                                  unsigned int label,
                                  unsigned int decrop,
                                  std::vector<int> loind,
                                  std::vector<int> upind  )
{
  typedef typename ImageType::Pointer ImagePointerType;

  ImagePointerType antsimage1 = in_image1.ptr;
  ImagePointerType antsimage2 = in_image2.ptr; 

  if ( decrop == 0 )
  {
    ImagePointerType out_image = cropImageHelper<ImageType>(antsimage1, antsimage2, label);
    AntsImage<ImageType> out_ants_image = { out_image };
    return out_ants_image;
  }
  else if ( decrop == 1 )
  {
    ImagePointerType out_image = decropImageHelper<ImageType>(antsimage1, antsimage2);
    AntsImage<ImageType> out_ants_image = { out_image };
    return out_ants_image;
  }

  else if ( decrop == 2 )
  {
    ImagePointerType out_image = cropIndHelper<ImageType>(antsimage1, loind, upind);
    AntsImage<ImageType> out_ants_image = { out_image };
    return out_ants_image;
  }

  ImagePointerType out_image = ImageType::New();
  AntsImage<ImageType> out_ants_image = { out_image };
  return out_ants_image;

}

void local_cropImage(nb::module_ &m) {
  m.def("cropImage", &cropImage<itk::Image<float, 2>>);
  m.def("cropImage", &cropImage<itk::Image<float, 3>>);
}
