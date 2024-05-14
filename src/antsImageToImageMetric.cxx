
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <vector>
#include <string>

#include "antscore/antsUtilities.h"
#include "itkDisplacementFieldTransform.h"
#include "itkImageToImageMetricv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkANTSNeighborhoodCorrelationImageToImageMetricv4.h"
#include "itkDemonsImageToImageMetricv4.h"
#include "itkJointHistogramMutualInformationImageToImageMetricv4.h"
#include "itkImageMaskSpatialObject.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageRandomConstIteratorWithIndex.h"

#include "antsImageToImageMetric.h"

namespace nb = nanobind;
using namespace nb::literals;


template <typename MetricType, unsigned int Dimension>
void wrapANTsImageToImageMetric(nb::module_ & m, std::string const & suffix) {
    nb::class_<ANTsImageToImageMetric<MetricType>>(m, ("ANTsImageToImageMetric" + suffix).c_str())
        //.def(py::init<>())

        // read only properties
        .def_ro("precision", &ANTsImageToImageMetric<MetricType>::precision)
        .def_ro("metrictype", &ANTsImageToImageMetric<MetricType>::metrictype)
        .def_ro("dimension", &ANTsImageToImageMetric<MetricType>::dimension)
        .def_ro("isVector", &ANTsImageToImageMetric<MetricType>::isVector)
        .def_ro("pointer", &ANTsImageToImageMetric<MetricType>::pointer)

        .def("setFixedImage", &ANTsImageToImageMetric<MetricType>::template setFixedImage<itk::Image<float, Dimension>>)
        .def("setMovingImage", &ANTsImageToImageMetric<MetricType>::template setMovingImage<itk::Image<float, Dimension>>)
        .def("setSampling", &ANTsImageToImageMetric<MetricType>::setSampling)
        .def("initialize", &ANTsImageToImageMetric<MetricType>::initialize)
        .def("getValue", &ANTsImageToImageMetric<MetricType>::getValue);

}

void local_antsImageToImageMetric(nb::module_ &m) {
    wrapANTsImageToImageMetric<itk::ImageToImageMetricv4<itk::Image<float, 2>,itk::Image<float,2>>,2>(m, "F2");
    wrapANTsImageToImageMetric<itk::ImageToImageMetricv4<itk::Image<float, 3>,itk::Image<float,3>>,3>(m, "F3");

    m.def("new_ants_metricF2", &new_ants_metric<itk::ImageToImageMetricv4<itk::Image<float, 2>,itk::Image<float,2>>,2>);
    m.def("new_ants_metricF3", &new_ants_metric<itk::ImageToImageMetricv4<itk::Image<float, 3>,itk::Image<float,3>>,3>);

    m.def("create_ants_metricF2", &create_ants_metric<itk::ImageToImageMetricv4<itk::Image<float, 2>,itk::Image<float,2>>,2>);
    m.def("create_ants_metricF3", &create_ants_metric<itk::ImageToImageMetricv4<itk::Image<float, 3>,itk::Image<float,3>>,3>);
}



