#include <nanobind/nanobind.h>

#include "local_antsImage.cpp"
#include "local_antsImageClone.cpp"
#include "local_antsImageHeaderInfo.cpp"
#include "local_test.cpp"
#include "local_imageRead.cpp"
#include "wrap_antsAffineInitializer.cxx"
#include "wrap_Atropos.cxx"
#include "wrap_antsRegistration.cxx"
#include "wrap_iMath.cxx"
#include "wrap_ThresholdImage.cxx"

namespace nb = nanobind;

void local_antsImage(nb::module_ &);
void local_antsImageClone(nb::module_ &);
void local_antsImageHeaderInfo(nb::module_ &);
void local_test(nb::module_ &);
void local_imageRead(nb::module_ &);
void wrap_antsAffineInitializer(nb::module_ &);
void wrap_Atropos(nb::module_ &);
void wrap_antsRegistration(nb::module_ &);
void wrap_iMath(nb::module_ &);
void wrap_ThresholdImage(nb::module_ &);
/* ... */

NB_MODULE(lib, m) {
    local_antsImage(m);
    local_antsImageClone(m);
    local_antsImageHeaderInfo(m);
    local_test(m);
    local_imageRead(m);
    wrap_antsAffineInitializer(m);
    wrap_Atropos(m);
    wrap_antsRegistration(m);
    wrap_iMath(m);
    wrap_ThresholdImage(m);
    /* ... */
}