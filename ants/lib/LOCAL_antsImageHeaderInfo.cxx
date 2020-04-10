
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileReader.h"

namespace py = pybind11;

using namespace py::literals;


py::dict antsImageHeaderInfo( std::string fname )
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

  return py::dict("pixelclass"_a=pixelClass,
                  "pixeltype"_a=pixeltype,
                  "nDimensions"_a=numDimensions,
                  "nComponents"_a=numComponents,
                  "dimensions"_a=dimensions,
                  "spacing"_a=spacing,
                  "origin"_a=origin,
                  "direction"_a=direction);
}


PYBIND11_MODULE(antsImageHeaderInfo, m)
{
  m.def("antsImageHeaderInfo", &antsImageHeaderInfo);
}
