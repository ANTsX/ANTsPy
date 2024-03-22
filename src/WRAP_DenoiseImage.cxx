
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/DenoiseImage.h"

namespace py = pybind11;

int DenoiseImage( std::vector<std::string> instring )
{
    return ants::DenoiseImage(instring, NULL);
}

PYBIND11_MODULE(DenoiseImage, m)
{
  m.def("DenoiseImage", &DenoiseImage);
}