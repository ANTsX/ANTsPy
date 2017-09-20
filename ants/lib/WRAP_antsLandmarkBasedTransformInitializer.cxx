
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antscore/antsLandmarkBasedTransformInitializer.h"

namespace py = pybind11;

int antsLandmarkBasedTransformInitializer( std::vector<std::string> instring )
{
    return ants::antsLandmarkBasedTransformInitializer(instring, NULL);
}

PYBIND11_MODULE(antsLandmarkBasedTransformInitializer, m)
{
  m.def("antsLandmarkBasedTransformInitializer", &antsLandmarkBasedTransformInitializer);
}