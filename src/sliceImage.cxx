
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include "itkImage.h"
#include "itkExtractImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template < typename ImageType, typename SliceImageType>
AntsImage<SliceImageType> sliceImage( AntsImage<ImageType> & antsImage, int plane, int slice, int collapseStrategy)
{
    typename ImageType::Pointer itkImage = antsImage.ptr;

    typedef itk::ExtractImageFilter< ImageType, SliceImageType > FilterType;
    typename FilterType::Pointer filter = FilterType::New();

    typename ImageType::RegionType inputRegion = itkImage->GetLargestPossibleRegion();
    typename ImageType::SizeType size = inputRegion.GetSize();
    size[plane] = 0;

    typename ImageType::IndexType start = inputRegion.GetIndex();
    const unsigned int sliceNumber = slice;
    start[plane] = sliceNumber;

    typename ImageType::RegionType desiredRegion;
    desiredRegion.SetSize( size );
    desiredRegion.SetIndex( start );

    filter->SetExtractionRegion( desiredRegion );
    filter->SetInput( itkImage );
    if( collapseStrategy == 0 )
      { 
      filter->SetDirectionCollapseToSubmatrix();
      } 
    else if( collapseStrategy == 1 )
      {
      filter->SetDirectionCollapseToIdentity();
      } 
    else // if( collapseStrategy == 2 ) 
      {
      filter->SetDirectionCollapseToGuess();
      } 
  
    filter->Update();

    AntsImage<SliceImageType> myImage = { filter->GetOutput() };
    return myImage;

}

void local_sliceImage(nb::module_ &m) 
{
    m.def("sliceImage", &sliceImage<itk::Image<float,3>, itk::Image<float,2>>);
    m.def("sliceImage", &sliceImage<itk::Image<float,4>, itk::Image<float,3>>);
}
