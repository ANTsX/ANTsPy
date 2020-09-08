
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/ThresholdImage.h"

namespace py = pybind11;

int ThresholdImage( std::vector<std::string> instring )
{
    return ants::ThresholdImage(instring, NULL);
}

PYBIND11_MODULE(ThresholdImage, m)
{
  m.def("ThresholdImage", &ThresholdImage);
}