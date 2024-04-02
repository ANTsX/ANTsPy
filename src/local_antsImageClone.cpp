#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"

namespace nb = nanobind;

using namespace nb::literals;

template<typename InImageType, typename OutImageType>
typename OutImageType::Pointer antsImageCloneHelper( typename InImageType::Pointer in_image )
{

  typename OutImageType::Pointer out_image = OutImageType::New() ;
  out_image->SetRegions( in_image->GetLargestPossibleRegion() ) ;
  out_image->SetSpacing( in_image->GetSpacing() ) ;
  out_image->SetOrigin( in_image->GetOrigin() ) ;
  out_image->SetDirection( in_image->GetDirection() );
  out_image->Allocate() ;

  itk::ImageRegionConstIterator< InImageType > in_iterator( in_image , in_image->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< OutImageType > out_iterator( out_image , out_image->GetLargestPossibleRegion() ) ;
  for( in_iterator.GoToBegin() , out_iterator.GoToBegin() ; !in_iterator.IsAtEnd() ; ++in_iterator , ++out_iterator )
    {
    out_iterator.Set( static_cast< typename OutImageType::PixelType >( in_iterator.Get() ) ) ;
    }

    //typedef typename OutImageType::Pointer ImagePointerType;
    //ImagePointerType * ptr = new ImagePointerType( out_image );
    return out_image;

}

template< typename ImageType >
typename ImageType::Pointer antsImageClone(typename ImageType::Pointer ptr, std::string inType, std::string outType) {
    typename ImageType::Pointer itkImage = asImage<ImageType>( ptr );
    return antsImageCloneHelper<ImageType, ImageType>( itkImage );
}

void local_antsImageClone(nb::module_ &m) {
    m.def("antsImageClone", &antsImageClone<itk::Image<unsigned char, 3>>);
    m.def("antsImageClone", &antsImageClone<itk::Image<unsigned int, 3>>);
    m.def("antsImageClone", &antsImageClone<itk::Image<float, 3>>);
    m.def("antsImageClone", &antsImageClone<itk::Image<double, 3>>);
}


