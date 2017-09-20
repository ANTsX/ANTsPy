
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/TileImages.h"

namespace py = pybind11;

int TileImages( std::vector<std::string> instring )
{
    return ants::TileImages(instring, NULL);
}

PYBIND11_MODULE(TileImages, m)
{
  m.def("TileImages", &TileImages);
}