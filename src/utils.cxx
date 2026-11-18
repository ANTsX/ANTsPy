
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include "itkMultiThreaderBase.h"

namespace nb = nanobind;
using namespace nb::literals;


void setGlobalDefaultNumberOfThreads(unsigned int numThreads)
{
    itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(numThreads);
}

unsigned int getGlobalDefaultNumberOfThreads()
{
    return itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads();
}

void local_utils(nb::module_ &m)
{
    m.def("setGlobalDefaultNumberOfThreads", &setGlobalDefaultNumberOfThreads);
    m.def("getGlobalDefaultNumberOfThreads", &getGlobalDefaultNumberOfThreads);
}
