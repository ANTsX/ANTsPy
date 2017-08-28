
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
  const unsigned int pixelCode = imageIO->GetComponentType();

  std::vector<double> dimensions( numDimensions );
  std::vector<double> spacing( numDimensions );
  std::vector<double> origin( numDimensions );
  std::vector<std::vector<double> > direction( numDimensions, std::vector<double>(numDimensions) );

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
    case 0: // UNKNOWNCOMPONENTTYPE - exception here?
      pixeltype = "unknown";
      break;
    case 1: // UCHAR
      pixeltype = "unsigned char";
      break;
    case 2: // CHAR
      pixeltype = "char";
      break;
    case 3: // USHORT
      pixeltype = "unsigned short";
      break;
    case 4: // SHORT
      pixeltype = "short";
      break;
    case 5: // UINT
      pixeltype = "unsigned int";
      break;
    case 6: // INT
      pixeltype = "int";
      break;
    case 7: // ULONG
      pixeltype = "unsigned long";
      break;
    case 8: // LONG
      pixeltype = "long";
      break;
    case 9: // FLOAT
      pixeltype = "float";
      break;
    case 10: // DOUBLE
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
