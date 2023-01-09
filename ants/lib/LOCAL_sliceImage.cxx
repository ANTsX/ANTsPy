
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "itkImage.h"
#include "itkExtractImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template < typename ImageType, typename PixelType, unsigned int NewDimension >
py::capsule sliceImage( py::capsule antsImage, int plane, int slice, int collapseStrategy)
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType itkImage = as< ImageType >( antsImage );

    typedef itk::Image<PixelType, NewDimension> SliceImageType;
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

    return wrap<SliceImageType>( filter->GetOutput() );

}

PYBIND11_MODULE(sliceImage, m)
{
    m.def("sliceImageF3", &sliceImage<itk::Image<float,3>, float, 2>);
    m.def("sliceImageF4", &sliceImage<itk::Image<float,4>, float, 3>);
}
