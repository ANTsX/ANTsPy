
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template<typename InImageType, typename OutImageType>
py::capsule antsImageClone( py::capsule antsImage )
{
  typedef typename InImageType::Pointer InImagePointerType;
  InImagePointerType in_image = as< InImageType >( antsImage );

  typename OutImageType::Pointer out_image = OutImageType::New() ;
  out_image->SetRegions( in_image->GetLargestPossibleRegion() ) ;
  out_image->SetSpacing( in_image->GetSpacing() ) ;
  out_image->SetOrigin( in_image->GetOrigin() ) ;
  out_image->SetDirection( in_image->GetDirection() );
  //out_image->CopyInformation( in_image );
  out_image->Allocate() ;

  itk::ImageRegionConstIterator< InImageType > in_iterator( in_image , in_image->GetLargestPossibleRegion() ) ;
  itk::ImageRegionIterator< OutImageType > out_iterator( out_image , out_image->GetLargestPossibleRegion() ) ;
  for( in_iterator.GoToBegin() , out_iterator.GoToBegin() ; !in_iterator.IsAtEnd() ; ++in_iterator , ++out_iterator )
    {
    out_iterator.Set( static_cast< typename OutImageType::PixelType >( in_iterator.Get() ) ) ;
    }
  return wrap< OutImageType >( out_image );
}

template <typename InImageType, typename OutImageType>
void wrapantsImageClone(py::module & m, std::string const & suffix) {
  m.def(("antsImageClone"+suffix).c_str(), &antsImageClone<InImageType,OutImageType>);
}

//pixels: Image, VectorImage
//types: unsigned char, unsigned int, float, double
//dims: 2, 3, 4
PYBIND11_MODULE(antsImageClone, m)
{
  // dim = 2
  wrapantsImageClone<itk::Image<unsigned char,2>,itk::Image<unsigned char,2>>(m, "UC2UC2");
  wrapantsImageClone<itk::Image<unsigned char,2>,itk::Image<unsigned int,2>>(m, "UC2UI2");
  wrapantsImageClone<itk::Image<unsigned char,2>,itk::Image<float,2>>(m, "UC2F2");

  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned char,2>>(m, "UI2UC2");
  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned int,2>>(m, "UI2UI2");
  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<float,2>>(m, "UI2F2");

  wrapantsImageClone<itk::Image<float,2>,itk::Image<unsigned char,2>>(m, "F2UC2");
  wrapantsImageClone<itk::Image<float,2>,itk::Image<unsigned int,2>>(m, "F2UI2");
  wrapantsImageClone<itk::Image<float,2>,itk::Image<float,2>>(m, "F2F2");

  // dim = 3
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned char,3>>(m, "UC3UC3");
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned int,3>>(m, "UC3UI3");
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<float,3>>(m, "UC3F3");

  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned char,3>>(m, "UI3UC3");
  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned int,3>>(m, "UI3UI3");
  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<float,3>>(m, "UI3F3");

  wrapantsImageClone<itk::Image<float,3>,itk::Image<unsigned char,3>>(m, "F3UC3");
  wrapantsImageClone<itk::Image<float,3>,itk::Image<unsigned int,3>>(m, "F3UI3");
  wrapantsImageClone<itk::Image<float,3>,itk::Image<float,3>>(m, "F3F3");

  // dim = 4
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned char,4>>(m, "UC4UC4");
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned int,4>>(m, "UC4UI4");
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<float,4>>(m, "UC4F4");

  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned char,4>>(m, "UI4UC4");
  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned int,4>>(m, "UI4UI4");
  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<float,4>>(m, "UI4F4");

  wrapantsImageClone<itk::Image<float,4>,itk::Image<unsigned char,4>>(m, "F4UC4");
  wrapantsImageClone<itk::Image<float,4>,itk::Image<unsigned int,4>>(m, "F4UI4");
  wrapantsImageClone<itk::Image<float,4>,itk::Image<float,4>>(m, "F4F4");

}


