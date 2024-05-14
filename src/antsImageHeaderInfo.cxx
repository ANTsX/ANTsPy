
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileReader.h"

namespace nb = nanobind;
using namespace nb::literals;


nb::dict antsImageHeaderInfo( std::string fname )
{

  itk::ImageIOBase::Pointer imageIO =
      itk::ImageIOFactory::CreateImageIO(
          fname.c_str(), itk::ImageIOFactory::ReadMode);

  imageIO->SetFileName(fname);
  imageIO->ReadImageInformation();

  const size_t numDimensions =  imageIO->GetNumberOfDimensions();
  const size_t numComponents = imageIO->GetNumberOfComponents();
  const std::string pixelClass( imageIO->GetPixelTypeAsString(imageIO->GetPixelType()) );
  itk::ImageIOBase::IOComponentEnum pixelCode = imageIO->GetComponentType();

  std::vector<float> dimensions( numDimensions );
  std::vector<float> spacing( numDimensions );
  std::vector<float> origin( numDimensions );
  std::vector<std::vector<float> > direction( numDimensions, std::vector<float>(numDimensions) );

  for (unsigned int i=0; i<numDimensions; i++)
    {
    dimensions[i] = imageIO->GetDimensions(i);
    spacing[i] = imageIO->GetSpacing(i);
    origin[i] = imageIO->GetOrigin(i);
    for (unsigned int j=0; j<numDimensions; j++)
      {
      direction[i][j] = imageIO->GetDirection(i)[j];
      }
    }

  std::string pixeltype = "unknown";

  switch( pixelCode )
    {
    case itk::ImageIOBase::IOComponentType::UNKNOWNCOMPONENTTYPE: // UNKNOWNCOMPONENTTYPE - exception here?
      pixeltype = "unknown";
      break;
    case itk::ImageIOBase::IOComponentType::UCHAR: // UCHAR
      pixeltype = "unsigned char";
      break;
    case itk::ImageIOBase::IOComponentType::CHAR: // CHAR
      pixeltype = "char";
      break;
    case itk::ImageIOBase::IOComponentType::USHORT: // USHORT
      pixeltype = "unsigned short";
      break;
    case itk::ImageIOBase::IOComponentType::SHORT: // SHORT
      pixeltype = "short";
      break;
    case itk::ImageIOBase::IOComponentType::UINT: // UINT
      pixeltype = "unsigned int";
      break;
    case itk::ImageIOBase::IOComponentType::INT: // INT
      pixeltype = "int";
      break;
    case itk::ImageIOBase::IOComponentType::ULONG: // ULONG
      pixeltype = "unsigned long";
      break;
    case itk::ImageIOBase::IOComponentType::LONG: // LONG
      pixeltype = "long";
      break;
    case itk::ImageIOBase::IOComponentType::ULONGLONG: // LONGLONG
      pixeltype = "ulonglong";
      break;
    case itk::ImageIOBase::IOComponentType::LONGLONG: // LONGLONG
      pixeltype = "longlong";
      break;
    case itk::ImageIOBase::IOComponentType::FLOAT: // FLOAT
      pixeltype = "float";
      break;
    case itk::ImageIOBase::IOComponentType::DOUBLE: // DOUBLE
      pixeltype = "double";
      break;
    default:
      pixeltype = "invalid";
    }

  nb::dict info;
  info["pixelclass"] = pixelClass;
  info["pixeltype"] = pixeltype;
  info["nDimensions"] = numDimensions;
  info["nComponents"] = numComponents;
  info["dimensions"] = dimensions;
  info["spacing"] = spacing;
  info["origin"] = origin;
  info["direction"] = direction;

  return info;
}


void local_antsImageHeaderInfo(nb::module_ &m)
{
  m.def("antsImageHeaderInfo", &antsImageHeaderInfo);
}
