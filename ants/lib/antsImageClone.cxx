
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"

#include "antsImage.h"

namespace py = pybind11;

template<typename InImageType, typename OutImageType>
ANTsImage<OutImageType> antsImageClone( ANTsImage<InImageType> antsImage )
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
  wrapantsImageClone<itk::Image<unsigned char,2>,itk::Image<double,2>>(m, "UC2D2");

  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned char,2>>(m, "UI2UC2");
  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<unsigned int,2>>(m, "UI2UI2");
  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<float,2>>(m, "UI2F2");
  wrapantsImageClone<itk::Image<unsigned int,2>,itk::Image<double,2>>(m, "UI2D2");

  wrapantsImageClone<itk::Image<float,2>,itk::Image<unsigned char,2>>(m, "F2UC2");
  wrapantsImageClone<itk::Image<float,2>,itk::Image<unsigned int,2>>(m, "F2UI2");
  wrapantsImageClone<itk::Image<float,2>,itk::Image<float,2>>(m, "F2F2");
  wrapantsImageClone<itk::Image<float,2>,itk::Image<double,2>>(m, "F2D2");

  wrapantsImageClone<itk::Image<double,2>,itk::Image<unsigned char,2>>(m, "D2UC2");
  wrapantsImageClone<itk::Image<double,2>,itk::Image<unsigned int,2>>(m, "D2UI2");
  wrapantsImageClone<itk::Image<double,2>,itk::Image<float,2>>(m, "D2F2");
  wrapantsImageClone<itk::Image<double,2>,itk::Image<double,2>>(m, "D2D2");


  // dim = 3
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned char,3>>(m, "UC3UC2");
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<unsigned int,3>>(m, "UC3UI3");
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<float,3>>(m, "UC3F3");
  wrapantsImageClone<itk::Image<unsigned char,3>,itk::Image<double,3>>(m, "UC3D3");

  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned char,3>>(m, "UI3UC3");
  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<unsigned int,3>>(m, "UI3UI3");
  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<float,3>>(m, "UI3F3");
  wrapantsImageClone<itk::Image<unsigned int,3>,itk::Image<double,3>>(m, "UI3D3");

  wrapantsImageClone<itk::Image<float,3>,itk::Image<unsigned char,3>>(m, "F3UC3");
  wrapantsImageClone<itk::Image<float,3>,itk::Image<unsigned int,3>>(m, "F3UI3");
  wrapantsImageClone<itk::Image<float,3>,itk::Image<float,3>>(m, "F3F3");
  wrapantsImageClone<itk::Image<float,3>,itk::Image<double,3>>(m, "F3D3");

  wrapantsImageClone<itk::Image<double,3>,itk::Image<unsigned char,3>>(m, "D3UC3");
  wrapantsImageClone<itk::Image<double,3>,itk::Image<unsigned int,3>>(m, "D3UI3");
  wrapantsImageClone<itk::Image<double,3>,itk::Image<float,3>>(m, "D3F3");
  wrapantsImageClone<itk::Image<double,3>,itk::Image<double,3>>(m, "D3D3");

  // dim = 4
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned char,4>>(m, "UC4UC4");
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<unsigned int,4>>(m, "UC4UI4");
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<float,4>>(m, "UC4F4");
  wrapantsImageClone<itk::Image<unsigned char,4>,itk::Image<double,4>>(m, "UC4D4");

  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned char,4>>(m, "UI4UC4");
  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<unsigned int,4>>(m, "UI4UI4");
  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<float,4>>(m, "UI4F4");
  wrapantsImageClone<itk::Image<unsigned int,4>,itk::Image<double,4>>(m, "UI4D4");

  wrapantsImageClone<itk::Image<float,4>,itk::Image<unsigned char,4>>(m, "F4UC4");
  wrapantsImageClone<itk::Image<float,4>,itk::Image<unsigned int,4>>(m, "F4UI4");
  wrapantsImageClone<itk::Image<float,4>,itk::Image<float,4>>(m, "F4F4");
  wrapantsImageClone<itk::Image<float,4>,itk::Image<double,4>>(m, "F4D4");

  wrapantsImageClone<itk::Image<double,4>,itk::Image<unsigned char,4>>(m, "D4UC4");
  wrapantsImageClone<itk::Image<double,4>,itk::Image<unsigned int,4>>(m, "D4UI4");
  wrapantsImageClone<itk::Image<double,4>,itk::Image<float,4>>(m, "D4F4");
  wrapantsImageClone<itk::Image<double,4>,itk::Image<double,4>>(m, "D4D4");

  // VectorImage
  // dim = 2
  wrapantsImageClone<itk::VectorImage<unsigned char,2>,itk::VectorImage<unsigned char,2>>(m, "VUC2VUC2");
  wrapantsImageClone<itk::VectorImage<unsigned char,2>,itk::VectorImage<unsigned int,2>>(m, "VUC2VUI2");
  wrapantsImageClone<itk::VectorImage<unsigned char,2>,itk::VectorImage<float,2>>(m, "VUC2VF2");
  wrapantsImageClone<itk::VectorImage<unsigned char,2>,itk::VectorImage<double,2>>(m, "VUC2VD2");

  wrapantsImageClone<itk::VectorImage<unsigned int,2>,itk::VectorImage<unsigned char,2>>(m, "VUI2VUC2");
  wrapantsImageClone<itk::VectorImage<unsigned int,2>,itk::VectorImage<unsigned int,2>>(m, "VUI2VUI2");
  wrapantsImageClone<itk::VectorImage<unsigned int,2>,itk::VectorImage<float,2>>(m, "VUI2VF2");
  wrapantsImageClone<itk::VectorImage<unsigned int,2>,itk::VectorImage<double,2>>(m, "VUI2VD2");

  wrapantsImageClone<itk::VectorImage<float,2>,itk::VectorImage<unsigned char,2>>(m, "VF2VUC2");
  wrapantsImageClone<itk::VectorImage<float,2>,itk::VectorImage<unsigned int,2>>(m, "VF2VUI2");
  wrapantsImageClone<itk::VectorImage<float,2>,itk::VectorImage<float,2>>(m, "VF2VF2");
  wrapantsImageClone<itk::VectorImage<float,2>,itk::VectorImage<double,2>>(m, "VF2VD2");

  wrapantsImageClone<itk::VectorImage<double,2>,itk::VectorImage<unsigned char,2>>(m, "VD2VUC2");
  wrapantsImageClone<itk::VectorImage<double,2>,itk::VectorImage<unsigned int,2>>(m, "VD2VUI2");
  wrapantsImageClone<itk::VectorImage<double,2>,itk::VectorImage<float,2>>(m, "VD2VF2");
  wrapantsImageClone<itk::VectorImage<double,2>,itk::VectorImage<double,2>>(m, "VD2VD2");


  // dim = 3
  wrapantsImageClone<itk::VectorImage<unsigned char,3>,itk::VectorImage<unsigned char,3>>(m, "VUC3VUC2");
  wrapantsImageClone<itk::VectorImage<unsigned char,3>,itk::VectorImage<unsigned int,3>>(m, "VUC3VUI3");
  wrapantsImageClone<itk::VectorImage<unsigned char,3>,itk::VectorImage<float,3>>(m, "VUC3VF3");
  wrapantsImageClone<itk::VectorImage<unsigned char,3>,itk::VectorImage<double,3>>(m, "VUC3VD3");

  wrapantsImageClone<itk::VectorImage<unsigned int,3>,itk::VectorImage<unsigned char,3>>(m, "VUI3VUC3");
  wrapantsImageClone<itk::VectorImage<unsigned int,3>,itk::VectorImage<unsigned int,3>>(m, "VUI3VUI3");
  wrapantsImageClone<itk::VectorImage<unsigned int,3>,itk::VectorImage<float,3>>(m, "VUI3VF3");
  wrapantsImageClone<itk::VectorImage<unsigned int,3>,itk::VectorImage<double,3>>(m, "VUI3VD3");

  wrapantsImageClone<itk::VectorImage<float,3>,itk::VectorImage<unsigned char,3>>(m, "VF3VUC3");
  wrapantsImageClone<itk::VectorImage<float,3>,itk::VectorImage<unsigned int,3>>(m, "VF3VUI3");
  wrapantsImageClone<itk::VectorImage<float,3>,itk::VectorImage<float,3>>(m, "VF3VF3");
  wrapantsImageClone<itk::VectorImage<float,3>,itk::VectorImage<double,3>>(m, "VF3VD3");

  wrapantsImageClone<itk::VectorImage<double,3>,itk::VectorImage<unsigned char,3>>(m, "VD3VUC3");
  wrapantsImageClone<itk::VectorImage<double,3>,itk::VectorImage<unsigned int,3>>(m, "VD3VUI3");
  wrapantsImageClone<itk::VectorImage<double,3>,itk::VectorImage<float,3>>(m, "VD3VF3");
  wrapantsImageClone<itk::VectorImage<double,3>,itk::VectorImage<double,3>>(m, "VD3VD3");

  // dim = 4
  wrapantsImageClone<itk::VectorImage<unsigned char,4>,itk::VectorImage<unsigned char,4>>(m, "VUC4VUC4");
  wrapantsImageClone<itk::VectorImage<unsigned char,4>,itk::VectorImage<unsigned int,4>>(m, "VUC4VUI4");
  wrapantsImageClone<itk::VectorImage<unsigned char,4>,itk::VectorImage<float,4>>(m, "VUC4VF4");
  wrapantsImageClone<itk::VectorImage<unsigned char,4>,itk::VectorImage<double,4>>(m, "VUC4VD4");

  wrapantsImageClone<itk::VectorImage<unsigned int,4>,itk::VectorImage<unsigned char,4>>(m, "VUI4VUC4");
  wrapantsImageClone<itk::VectorImage<unsigned int,4>,itk::VectorImage<unsigned int,4>>(m, "VUI4VUI4");
  wrapantsImageClone<itk::VectorImage<unsigned int,4>,itk::VectorImage<float,4>>(m, "VUI4VF4");
  wrapantsImageClone<itk::VectorImage<unsigned int,4>,itk::VectorImage<double,4>>(m, "VUI4VD4");

  wrapantsImageClone<itk::VectorImage<float,4>,itk::VectorImage<unsigned char,4>>(m, "VF4VUC4");
  wrapantsImageClone<itk::VectorImage<float,4>,itk::VectorImage<unsigned int,4>>(m, "VF4VUI4");
  wrapantsImageClone<itk::VectorImage<float,4>,itk::VectorImage<float,4>>(m, "VF4VF4");
  wrapantsImageClone<itk::VectorImage<float,4>,itk::VectorImage<double,4>>(m, "VF4VD4");

  wrapantsImageClone<itk::VectorImage<double,4>,itk::VectorImage<unsigned char,4>>(m, "VD4VUC4");
  wrapantsImageClone<itk::VectorImage<double,4>,itk::VectorImage<unsigned int,4>>(m, "VD4VUI4");
  wrapantsImageClone<itk::VectorImage<double,4>,itk::VectorImage<float,4>>(m, "VD4VF4");
  wrapantsImageClone<itk::VectorImage<double,4>,itk::VectorImage<double,4>>(m, "VD4VD4");

}


