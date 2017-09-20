
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/TileImage.h"

namespace py = pybind11;

int TileImage( std::vector<std::string> instring )
{
    return ants::TileImage(instring, NULL);
}

PYBIND11_MODULE(TileImage, m)
{
  m.def("TileImage", &TileImage);
}