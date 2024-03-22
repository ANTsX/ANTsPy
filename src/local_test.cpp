#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "itkImageIOBase.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include <iostream>

#include "antscore/DenoiseImage.h"

#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;

using namespace nb::literals;

using StrVector = std::vector<std::string>;


void * testfn()
{
  using ImageType = itk::Image<unsigned char, 3>;
  auto image = ImageType::New();

  ImageType::IndexType start;
  start[0] = 0; // first index on X
  start[1] = 0; // first index on Y
  start[2] = 0; // first index on Z

  ImageType::SizeType size;
  size[0] = 200; // size along X
  size[1] = 200; // size along Y
  size[2] = 200; // size along Z

  ImageType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  image->SetRegions(region);
  image->Allocate();

  std::cout << image << std::endl;

  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType * ptr = new ImagePointerType( image );

  return ptr;
}


void local_test(nb::module_ &m) {
    m.def("testfn", &testfn);
}
