
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/KellyKapowski.h"

namespace py = pybind11;

int KellyKapowski( std::vector<std::string> instring )
{
    return ants::KellyKapowski(instring, NULL);
}

PYBIND11_MODULE(KellyKapowski, m)
{
  m.def("KellyKapowski", &KellyKapowski);
}