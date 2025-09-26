#ifndef ANTS_READ_IMAGE_METADATA_H
#define ANTS_READ_IMAGE_METADATA_H

#include <string>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

nb::dict readImageMetadata(const std::string & path);

#endif // ANTS_READ_IMAGE_METADATA_H
