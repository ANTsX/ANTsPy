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
void * antsImageCloneHelper( typename InImageType::Pointer in_image )
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

    typedef typename OutImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( out_image );
    return ptr;

}

void * antsImageClone(void * ptr, std::string inType, std::string outType) {
    if (inType == "UC3") {
        void * antsImage;
        using ImageType = itk::Image<unsigned char, 3>;

        typedef typename ImageType::Pointer ImagePointerType;
        ImagePointerType itkImage = ImageType::New();
        typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
        itkImage = *real;

        return antsImageCloneHelper<ImageType, ImageType>( itkImage );
    }

    if (inType == "UI3") {
        void * antsImage;
        using ImageType = itk::Image<unsigned int, 3>;
        
        typedef typename ImageType::Pointer ImagePointerType;
        ImagePointerType itkImage = ImageType::New();
        typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
        itkImage = *real;

        return antsImageCloneHelper<ImageType, ImageType>( itkImage );
    }

    if (inType == "F3") {
        void * antsImage;
        using ImageType = itk::Image<float, 3>;
        
        typedef typename ImageType::Pointer ImagePointerType;
        ImagePointerType itkImage = ImageType::New();
        typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
        itkImage = *real;
        
        return antsImageCloneHelper<ImageType, ImageType>( itkImage );
    }

    if (inType == "D3") {
        void * antsImage;
        using ImageType = itk::Image<double, 3>;

        typedef typename ImageType::Pointer ImagePointerType;
        ImagePointerType itkImage = ImageType::New();
        typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
        itkImage = *real;
        
        return antsImageCloneHelper<ImageType, ImageType>( itkImage );
    }

}

void local_antsImageClone(nb::module_ &m) {
    m.def("antsImageClone", &antsImageClone);
}


