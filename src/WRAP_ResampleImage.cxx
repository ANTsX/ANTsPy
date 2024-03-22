
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/ResampleImage.h"

namespace py = pybind11;

int ResampleImage( std::vector<std::string> instring )
{
    return ants::ResampleImage(instring, NULL);
}

PYBIND11_MODULE(ResampleImage, m)
{
  m.def("ResampleImage", &ResampleImage);
}