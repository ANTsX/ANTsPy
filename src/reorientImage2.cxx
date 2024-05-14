

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

#include "itkImage.h"
#include "itkOrientImageFilter.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
AntsImage<ImageType> reorientImage2( AntsImage<ImageType> & antsImage , std::string newOrientation )
{
  typename ImageType::Pointer itkImage = antsImage.ptr;

  typename itk::OrientImageFilter<ImageType,ImageType>::Pointer orienter = itk::OrientImageFilter<ImageType,ImageType>::New();
  orienter->UseImageDirectionOn();

  if (newOrientation == "RIP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP);
  }
  else if (newOrientation == "LIP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIP);
  }
  else if (newOrientation == "RSP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP);
  }
  else if (newOrientation == "LSP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP);
  }
  else if (newOrientation == "RIA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA);
  }
  else if (newOrientation == "LIA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA);
  }
  else if (newOrientation == "RSA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA);
  }
  else if (newOrientation == "LSA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA);
  }
  else if (newOrientation == "IRP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP);
  }
  else if (newOrientation == "ILP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP);
  }
  else if (newOrientation == "SRP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP);
  }
  else if (newOrientation == "SLP") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP);
  }
  else if (newOrientation == "IRA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA);
  }
  else if (newOrientation == "ILA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA);
  }
  else if (newOrientation == "SRA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA);
  }
  else if (newOrientation == "SLA") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA);
  }
  else if (newOrientation == "RPI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI);
  }
  else if (newOrientation == "LPI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI);
  }
  else if (newOrientation == "RAI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI);
  }
  else if (newOrientation == "LAI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI);
  }
  else if (newOrientation == "RPS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS);
  }
  else if (newOrientation == "LPS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS);
  }
  else if (newOrientation == "RAS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
  }
  else if (newOrientation == "LAS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS);
  }
  else if (newOrientation == "PRI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI);
  }
  else if (newOrientation == "PLI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI);
  }
  else if (newOrientation == "ARI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI);
  }
  else if (newOrientation == "ALI") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI);
  }
  else if (newOrientation == "PRS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS);
  }
  else if (newOrientation == "PLS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS);
  }
  else if (newOrientation == "ARS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS);
  }
  else if (newOrientation == "ALS") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS);
  }
  else if (newOrientation == "IPR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR);
  }
  else if (newOrientation == "SPR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR);
  }
  else if (newOrientation == "IAR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR);
  }
  else if (newOrientation == "SAR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR);
  }
  else if (newOrientation == "IPL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL);
  }
  else if (newOrientation == "SPL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL);
  }
  else if (newOrientation == "IAL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL);
  }
  else if (newOrientation == "SAL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL);
  }
  else if (newOrientation == "PIR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR);
  }
  else if (newOrientation == "PSR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR);
  }
  else if (newOrientation == "AIR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR);
  }
  else if (newOrientation == "ASR") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR);
  }
  else if (newOrientation == "PIL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL);
  }
  else if (newOrientation == "PSL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL);
  }
  else if (newOrientation == "AIL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL);
  }
  else if (newOrientation == "ASL") {
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL);
  }
  orienter->SetInput( itkImage );
  orienter->Update();
  AntsImage<ImageType> myImage = { orienter->GetOutput() };
  return myImage;
}

void local_reorientImage2(nb::module_ &m) 
{
  m.def("reorientImage2", &reorientImage2<itk::Image<float,3>>);
}