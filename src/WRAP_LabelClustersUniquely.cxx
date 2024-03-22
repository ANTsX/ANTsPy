
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/LabelClustersUniquely.h"

namespace py = pybind11;

int LabelClustersUniquely( std::vector<std::string> instring )
{
    return ants::LabelClustersUniquely(instring, NULL);
}

PYBIND11_MODULE(LabelClustersUniquely, m)
{
  m.def("LabelClustersUniquely", &LabelClustersUniquely);
}