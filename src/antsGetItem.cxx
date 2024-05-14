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

template <typename ImageType, unsigned int ndim>
AntsImage<itk::Image<float, ndim>> getItem( AntsImage<ImageType> & antsImage, 
                              std::vector<unsigned long> starts, 
                              std::vector<unsigned long> sizes )
{
    typename ImageType::Pointer image = antsImage.ptr;

    using OutImageType = itk::Image<float, ndim>;

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
    m.def("getItem2",   &getItem<itk::Image<float,2>, 2>);
    m.def("getItem2",   &getItem<itk::Image<float,3>, 2>);
    m.def("getItem2",   &getItem<itk::Image<float,4>, 2>);
    m.def("getItem3",   &getItem<itk::Image<float,3>, 3>);
    m.def("getItem3",   &getItem<itk::Image<float,4>, 3>);
    m.def("getItem4",   &getItem<itk::Image<float,4>, 4>);
}