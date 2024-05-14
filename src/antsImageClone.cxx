
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<typename InImageType, typename OutImageType>
AntsImage<OutImageType> antsImageClone( AntsImage<InImageType> & myPointer )
{
  typename InImageType::Pointer in_image = myPointer.ptr;

  typename OutImageType::Pointer out_image = OutImageType::New() ;
  out_image->SetRegions( in_image->GetLargestPossibleRegion() ) ;
  out_image->SetSpacing( in_image->GetSpacing() ) ;
  out_image->SetOrigin( in_image->GetOrigin() ) ;
  out_image->SetDirection( in_image->GetDirection() );
  //out_image->CopyInformation( in_image );
  out_image->AllocateInitialized();

  itk::ImageRegionConstIterator< InImageType > in_iterator( in_image , in_image->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< OutImageType > out_iterator( out_image , out_image->GetLargestPossibleRegion() ) ;
  for( in_iterator.GoToBegin() , out_iterator.GoToBegin() ; !in_iterator.IsAtEnd() ; ++in_iterator , ++out_iterator )
    {
    out_iterator.Set( static_cast< typename OutImageType::PixelType >( in_iterator.Get() ) ) ;
    }
    AntsImage<OutImageType> outImage = { out_image };
    return outImage;
}
void local_antsImageClone(nb::module_ &m) {

    // call the function based on the image type you are converting TO.
    // the image type you are converting FROM should be automatically inferred by the template

    // dim = 2
    m.def("antsImageCloneUC2", &antsImageClone<itk::Image<unsigned char,2>,itk::Image<unsigned char,2>>);
    m.def("antsImageCloneUC2", &antsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned char,2>>);
    m.def("antsImageCloneUC2", &antsImageClone<itk::Image<float,2>,itk::Image<unsigned char,2>>);
    m.def("antsImageCloneUC2", &antsImageClone<itk::Image<double,2>,itk::Image<unsigned char,2>>);

    m.def("antsImageCloneUI2", &antsImageClone<itk::Image<unsigned char,2>,itk::Image<unsigned int,2>>);
    m.def("antsImageCloneUI2", &antsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned int,2>>);
    m.def("antsImageCloneUI2", &antsImageClone<itk::Image<float,2>,itk::Image<unsigned int,2>>);
    m.def("antsImageCloneUI2", &antsImageClone<itk::Image<double,2>,itk::Image<unsigned int,2>>);

    m.def("antsImageCloneF2", &antsImageClone<itk::Image<unsigned char,2>,itk::Image<float,2>>);
    m.def("antsImageCloneF2", &antsImageClone<itk::Image<unsigned int,2>,itk::Image<float,2>>);
    m.def("antsImageCloneF2", &antsImageClone<itk::Image<float,2>,itk::Image<float,2>>);
    m.def("antsImageCloneF2", &antsImageClone<itk::Image<double,2>,itk::Image<float,2>>);

    m.def("antsImageCloneD2", &antsImageClone<itk::Image<unsigned char,2>,itk::Image<double,2>>);
    m.def("antsImageCloneD2", &antsImageClone<itk::Image<unsigned int,2>,itk::Image<double,2>>);
    m.def("antsImageCloneD2", &antsImageClone<itk::Image<float,2>,itk::Image<double,2>>);
    m.def("antsImageCloneD2", &antsImageClone<itk::Image<double,2>,itk::Image<double,2>>);

    m.def("antsImageCloneRGBUC2", &antsImageClone<itk::Image<itk::RGBPixel<unsigned char>,2>,itk::Image<itk::RGBPixel<unsigned char>,2>>);

    // dim = 3

    m.def("antsImageCloneUC3", &antsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned char,3>>);
    m.def("antsImageCloneUC3", &antsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned char,3>>);
    m.def("antsImageCloneUC3", &antsImageClone<itk::Image<float,3>,itk::Image<unsigned char,3>>);
    m.def("antsImageCloneUC3", &antsImageClone<itk::Image<double,3>,itk::Image<unsigned char,3>>);

    m.def("antsImageCloneUI3", &antsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned int,3>>);
    m.def("antsImageCloneUI3", &antsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned int,3>>);
    m.def("antsImageCloneUI3", &antsImageClone<itk::Image<float,3>,itk::Image<unsigned int,3>>);
    m.def("antsImageCloneUI3", &antsImageClone<itk::Image<double,3>,itk::Image<unsigned int,3>>);

    m.def("antsImageCloneF3", &antsImageClone<itk::Image<unsigned char,3>,itk::Image<float,3>>);
    m.def("antsImageCloneF3", &antsImageClone<itk::Image<unsigned int,3>,itk::Image<float,3>>);
    m.def("antsImageCloneF3", &antsImageClone<itk::Image<float,3>,itk::Image<float,3>>);
    m.def("antsImageCloneF3", &antsImageClone<itk::Image<double,3>,itk::Image<float,3>>);

    m.def("antsImageCloneD3", &antsImageClone<itk::Image<unsigned char,3>,itk::Image<double,3>>);
    m.def("antsImageCloneD3", &antsImageClone<itk::Image<unsigned int,3>,itk::Image<double,3>>);
    m.def("antsImageCloneD3", &antsImageClone<itk::Image<float,3>,itk::Image<double,3>>);
    m.def("antsImageCloneD3", &antsImageClone<itk::Image<double,3>,itk::Image<double,3>>);

    m.def("antsImageCloneRGBUC3", &antsImageClone<itk::Image<itk::RGBPixel<unsigned char>,3>,itk::Image<itk::RGBPixel<unsigned char>,3>>);

    // dim = 4

    m.def("antsImageCloneUC4", &antsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned char,4>>);
    m.def("antsImageCloneUC4", &antsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned char,4>>);
    m.def("antsImageCloneUC4", &antsImageClone<itk::Image<float,4>,itk::Image<unsigned char,4>>);
    m.def("antsImageCloneUC4", &antsImageClone<itk::Image<double,4>,itk::Image<unsigned char,4>>);

    m.def("antsImageCloneUI4", &antsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned int,4>>);
    m.def("antsImageCloneUI4", &antsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned int,4>>);
    m.def("antsImageCloneUI4", &antsImageClone<itk::Image<float,4>,itk::Image<unsigned int,4>>);
    m.def("antsImageCloneUI4", &antsImageClone<itk::Image<double,4>,itk::Image<unsigned int,4>>);

    m.def("antsImageCloneF4", &antsImageClone<itk::Image<unsigned char,4>,itk::Image<float,4>>);
    m.def("antsImageCloneF4", &antsImageClone<itk::Image<unsigned int,4>,itk::Image<float,4>>);
    m.def("antsImageCloneF4", &antsImageClone<itk::Image<float,4>,itk::Image<float,4>>);
    m.def("antsImageCloneF4", &antsImageClone<itk::Image<double,4>,itk::Image<float,4>>);

    m.def("antsImageCloneD4", &antsImageClone<itk::Image<unsigned char,4>,itk::Image<double,4>>);
    m.def("antsImageCloneD4", &antsImageClone<itk::Image<unsigned int,4>,itk::Image<double,4>>);
    m.def("antsImageCloneD4", &antsImageClone<itk::Image<float,4>,itk::Image<double,4>>);
    m.def("antsImageCloneD4", &antsImageClone<itk::Image<double,4>,itk::Image<double,4>>);

    m.def("antsImageCloneRGBUC4", &antsImageClone<itk::Image<itk::RGBPixel<unsigned char>,4>,itk::Image<itk::RGBPixel<unsigned char>,4>>);
}