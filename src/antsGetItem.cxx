#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "itkImage.h"
#include <itkExtractImageFilter.h>

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType, class PixelType, unsigned int ndim>
AntsImage<itk::Image<PixelType, ndim>> getItem( AntsImage<ImageType> & antsImage, 
                              std::vector<unsigned long> starts, 
                              std::vector<unsigned long> sizes )
{
    typename ImageType::Pointer image = antsImage.ptr;

    using OutImageType = itk::Image<PixelType, ndim>;

    typename ImageType::IndexType desiredStart;
    typename ImageType::SizeType desiredSize;

    for( int i = 0 ; i < starts.size(); ++i )
    {
        desiredStart[i] = starts[i];
        desiredSize[i] = sizes[i];
    }

    typename ImageType::RegionType desiredRegion(desiredStart, desiredSize);

    using FilterType = itk::ExtractImageFilter<ImageType, OutImageType>;
    typename FilterType::Pointer filter = FilterType::New();
    filter->SetExtractionRegion(desiredRegion);
    filter->SetInput(image);
    filter->SetDirectionCollapseToIdentity(); // This is required.
    filter->Update();

    FixNonZeroIndex<OutImageType>( filter->GetOutput() );
    AntsImage<OutImageType> outImage = { filter->GetOutput() };
    return outImage;
}


void local_antsGetItem(nb::module_ &m) {
    m.def("getItem2",   &getItem<itk::Image<float,2>, float, 2>);
    m.def("getItem2",   &getItem<itk::Image<float,3>, float, 2>);  
    m.def("getItem2",   &getItem<itk::Image<float,4>, float, 2>);  
    m.def("getItem3",   &getItem<itk::Image<float,3>, float, 3>);  
    m.def("getItem3",   &getItem<itk::Image<float,4>, float, 3>);  
    m.def("getItem4",   &getItem<itk::Image<float,4>, float, 4>);  
    
    m.def("getItem2",   &getItem<itk::Image<unsigned char,2>, unsigned char, 2>);
    m.def("getItem2",   &getItem<itk::Image<unsigned char,3>, unsigned char, 2>);
    m.def("getItem2",   &getItem<itk::Image<unsigned char,4>, unsigned char, 2>);
    m.def("getItem3",   &getItem<itk::Image<unsigned char,3>, unsigned char, 3>);
    m.def("getItem3",   &getItem<itk::Image<unsigned char,4>, unsigned char, 3>);
    m.def("getItem4",   &getItem<itk::Image<unsigned char,4>, unsigned char, 4>);

    m.def("getItem2",   &getItem<itk::Image<unsigned int,2>, unsigned int, 2>);
    m.def("getItem2",   &getItem<itk::Image<unsigned int,3>, unsigned int, 2>);
    m.def("getItem2",   &getItem<itk::Image<unsigned int,4>, unsigned int, 2>);
    m.def("getItem3",   &getItem<itk::Image<unsigned int,3>, unsigned int, 3>);
    m.def("getItem3",   &getItem<itk::Image<unsigned int,4>, unsigned int, 3>);
    m.def("getItem4",   &getItem<itk::Image<unsigned int,4>, unsigned int, 4>);

    m.def("getItem2",   &getItem<itk::Image<double,2>, double, 2>);
    m.def("getItem2",   &getItem<itk::Image<double,3>, double, 2>);
    m.def("getItem2",   &getItem<itk::Image<double,4>, double, 2>);
    m.def("getItem3",   &getItem<itk::Image<double,3>, double, 3>);
    m.def("getItem3",   &getItem<itk::Image<double,4>, double, 3>);
    m.def("getItem4",   &getItem<itk::Image<double,4>, double, 4>);
}