
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/CreateTiledMosaic.h"

namespace py = pybind11;

int CreateTiledMosaic( std::vector<std::string> instring )
{
    return ants::CreateTiledMosaic(instring, NULL);
}

PYBIND11_MODULE(CreateTiledMosaic, m)
{
  m.def("CreateTiledMosaic", &CreateTiledMosaic);
}