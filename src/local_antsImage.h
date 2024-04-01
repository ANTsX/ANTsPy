#ifndef __ANTSPYIMAGE_H
#define __ANTSPYIMAGE_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>

#include "itkPyBuffer.h"
#include "itkImageIOBase.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include <iostream>

#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

using namespace nb::literals;

using StrVector = std::vector<std::string>;


template <typename ImageType>
typename ImageType::Pointer as( void * ptr )
{
    typename ImageType::Pointer * real  = static_cast<typename ImageType::Pointer *>(ptr); // static_cast or reinterpret_cast ??
    return *real;
}

// converts an ITK image pointer to a pointer
template <typename ImageType>
void * wrap( const typename ImageType::Pointer &image )
{
    typedef typename ImageType::Pointer ImagePointerType;
    ImagePointerType * ptr = new ImagePointerType( image );
    return ptr;
}

template <typename ImageType>
auto asImage( void * ptr ) {
        auto itkImage = ImageType::New();
        itkImage = as<ImageType>( ptr );
        return itkImage;
}

#endif