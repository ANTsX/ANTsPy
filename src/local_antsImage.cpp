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

#include "antscore/DenoiseImage.h"

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


template <typename ImageType>
auto asImage( void * ptr ) {
        auto itkImage = ImageType::New();
        itkImage = as<ImageType>( ptr );
        return itkImage;
}

void printfn(void * ptr) {
    using ImageType = itk::Image<unsigned char, 3>;
    auto image = asImage<ImageType>( ptr );
    std::cout << image << std::endl;
}


template <typename ImageType>
std::list<int> getShapeHelper( typename ImageType::Pointer image )
{
    unsigned int ndim = ImageType::GetImageDimension();
    image->UpdateOutputInformation();
    typename ImageType::SizeType shape = image->GetBufferedRegion().GetSize();
    std::list<int> shapelist;
    for (int i = 0; i < ndim; i++)
    {
        shapelist.push_back( shape[i] );
    }
    return shapelist;
}

std::list<int> getShape( void * ptr, std::string imageType )
{

    if (imageType == "UC3") {
        using ImageType = itk::Image<unsigned char, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getShapeHelper<ImageType>( itkImage );
    }

    if (imageType == "UI3") {
        using ImageType = itk::Image<unsigned int, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getShapeHelper<ImageType>( itkImage );
    }

    if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getShapeHelper<ImageType>( itkImage );
    }

    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getShapeHelper<ImageType>( itkImage );
    }

}


template <typename ImageType>
std::list<float> getSpacingHelper( typename ImageType::Pointer image )
{
    typename ImageType::SpacingType spacing = image->GetSpacing();
    unsigned int ndim = ImageType::GetImageDimension();

    std::list<float> spacinglist;
    for (int i = 0; i < ndim; i++)
    {
        spacinglist.push_back( spacing[i] );
    }

    return spacinglist;
}

std::list<float> getSpacing( void * ptr, std::string imageType )
{

    if (imageType == "UC3") {
        using ImageType = itk::Image<unsigned char, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getSpacingHelper<ImageType>( itkImage );
    }

    if (imageType == "UI3") {
        using ImageType = itk::Image<unsigned int, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getSpacingHelper<ImageType>( itkImage );
    }

    if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getSpacingHelper<ImageType>( itkImage );
    }

    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getSpacingHelper<ImageType>( itkImage );
    }
}

template <typename ImageType>
std::vector<double> getDirectionHelper( typename ImageType::Pointer image )
{
    typedef typename ImageType::DirectionType ImageDirectionType;
    ImageDirectionType direction = image->GetDirection();

    typedef typename ImageDirectionType::InternalMatrixType DirectionInternalMatrixType;
    DirectionInternalMatrixType fixed_matrix = direction.GetVnlMatrix();

    vnl_matrix<double> vnlmat1 = fixed_matrix.as_matrix();

    const unsigned int ndim = ImageType::SizeType::GetSizeDimension();

    std::vector<double> dvec;

    for (int i = 0; i < ndim; i++)
    {
        for (int j = 0; j < ndim; j++)
        {
            dvec.push_back(vnlmat1(i,j));
        }
    }
    return dvec;

}

std::vector<double> getDirection( void * ptr, std::string imageType )
{
    if (imageType == "UC3") {
        using ImageType = itk::Image<unsigned char, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getDirectionHelper<ImageType>( itkImage );
    }

    if (imageType == "UI3") {
        using ImageType = itk::Image<unsigned int, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getDirectionHelper<ImageType>( itkImage );
    }

    if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getDirectionHelper<ImageType>( itkImage );
    }

    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getDirectionHelper<ImageType>( itkImage );
    }
}


template <typename ImageType>
std::vector<double> getOriginHelper( typename ImageType::Pointer image )
{
    typename ImageType::PointType origin = image->GetOrigin();
    unsigned int ndim = ImageType::GetImageDimension();

    std::vector<double> originlist;
    for (int i = 0; i < ndim; i++)
    {
        originlist.push_back( origin[i] );
    }

    return originlist;
}

std::vector<double> getOrigin( void * ptr, std::string imageType )
{
    if (imageType == "UC3") {
        using ImageType = itk::Image<unsigned char, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getOriginHelper<ImageType>( itkImage );
    }

    if (imageType == "UI3") {
        using ImageType = itk::Image<unsigned int, 3>;
        auto itkImage = asImage<ImageType>( ptr );

        return getOriginHelper<ImageType>( itkImage );
    }

    if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getOriginHelper<ImageType>( itkImage );
    }

    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        
        return getOriginHelper<ImageType>( itkImage );
    }

}

std::string ptrstr(void * c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}

template <typename ImageType>
nb::object numpyHelperWorking( typename ImageType::Pointer itkImage )
{
    typedef itk::PyBuffer<ImageType> PyBufferType;
    PyObject * itkArray = PyBufferType::_GetArrayViewFromImage( itkImage );
    nb::object itkArrayObject = nb::steal( itkArray );
    //nb::ndarray<nb::numpy, float> itkArrayObject2 = nb::cast<nb::ndarray<nb::numpy, float>>( itkArray );
    return itkArrayObject;
}

template <typename ImageType>
nb::object numpyHelper( typename ImageType::Pointer itkImage )
{
    typedef itk::PyBuffer<ImageType> PyBufferType;
    PyObject * itkArray = PyBufferType::_GetArrayViewFromImage( itkImage );
    nb::object itkArrayObject = nb::steal( itkArray );
    return itkArrayObject;
}

nb::object toNumpy( void * ptr, std::string imageType )
{
    if (imageType == "D3") {
        using ImageType = itk::Image<double, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        return numpyHelper<ImageType>( itkImage );
    } else if (imageType == "F3") {
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        return numpyHelper<ImageType>( itkImage );
    } 
        using ImageType = itk::Image<float, 3>;
        auto itkImage = asImage<ImageType>( ptr );
        return numpyHelper<ImageType>( itkImage );
}

void local_antsImage(nb::module_ &m) {
    m.def("ptrstr", &ptrstr);
    m.def("toNumpy", &toNumpy);
    m.def("printfn", &printfn);
    m.def("getShape", &getShape);
    m.def("getSpacing", &getSpacing);
    m.def("getDirection", &getDirection);
    m.def("getOrigin", &getOrigin);
}

