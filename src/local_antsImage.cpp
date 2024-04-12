#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include "itkPyBuffer.h"
#include "itkImageIOBase.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageIOBase.h"
#include "itkNiftiImageIOFactory.h"
#include "itkMetaImageIOFactory.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include <iostream>

#include "local_antsImage.h"

namespace nb = nanobind;

using namespace nb::literals;

using StrVector = std::vector<std::string>;


template <typename ImageType>
std::list<int> getShape( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    unsigned int ndim = ImageType::GetImageDimension();
    image->UpdateOutputInformation();
    typename ImageType::SizeType shape = image->GetBufferedRegion().GetSize();
    std::list<int> shapelist;
    for (int i = 0; i < ndim; i++)
    {
        shapelist.push_back( shape[i] );
    }
    return shapelist;
}


template <typename ImageType>
nb::list getSpacing( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typename ImageType::SpacingType spacing = image->GetSpacing();
    unsigned int ndim = ImageType::GetImageDimension();

    nb::list spacinglist;
    for (int i = 0; i < ndim; i++)
    {
        spacinglist.append( spacing[i] );
    }

    return spacinglist;
}


template <typename ImageType>
void setSpacing( AntsImage<ImageType> & myPointer, std::vector<double> new_spacing)
{
    typename ImageType::Pointer itkImage = myPointer.ptr;
    unsigned int nvals = new_spacing.size();
    typename ImageType::SpacingType spacing = itkImage->GetSpacing();

    for (int i = 0; i < nvals; i++)
    {
        spacing[i] = new_spacing[i];
    }
    itkImage->SetSpacing( spacing );
}


template <typename ImageType>
std::vector<double> getDirection( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typedef typename ImageType::DirectionType ImageDirectionType;
    ImageDirectionType direction = image->GetDirection();

    typedef typename ImageDirectionType::InternalMatrixType DirectionInternalMatrixType;
    DirectionInternalMatrixType fixed_matrix = direction.GetVnlMatrix();

    vnl_matrix<double> vnlmat1 = fixed_matrix.as_matrix();

    const unsigned int ndim = ImageType::SizeType::GetSizeDimension();

    std::vector<double> dvec;

    for (int i = 0; i < ndim; i++)
    {
        for (int j = 0; j < ndim; j++)
        {
            dvec.push_back(vnlmat1(i,j));
        }
    }
    return dvec;

}


template <typename ImageType>
void setDirection( AntsImage<ImageType> & myPointer, std::vector<std::vector<double>> new_direction)
{

    typename ImageType::Pointer itkImage = myPointer.ptr;

    typename ImageType::DirectionType new_matrix2 = itkImage->GetDirection( );
    for ( std::size_t i = 0; i < new_direction.size(); i++ )
      for ( std::size_t j = 0; j < new_direction[0].size(); j++ ) {
        new_matrix2(i,j) = new_direction[i][j];
      }
    itkImage->SetDirection( new_matrix2 );
}


template <typename ImageType>
std::vector<double> getOrigin( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typename ImageType::PointType origin = image->GetOrigin();
    unsigned int ndim = ImageType::GetImageDimension();

    std::vector<double> originlist;
    for (int i = 0; i < ndim; i++)
    {
        originlist.push_back( origin[i] );
    }

    return originlist;
}

template <typename ImageType>
void setOrigin( AntsImage<ImageType> & myPointer, std::vector<double> new_origin)
{
    typename ImageType::Pointer itkImage = myPointer.ptr;
    unsigned int nvals = new_origin.size();
    typename ImageType::PointType origin = itkImage->GetOrigin();
    for (int i = 0; i < nvals; i++)
    {
        origin[i] = new_origin[i];
    }
    itkImage->SetOrigin( origin );
}

template <typename ImageType>
void toFile( AntsImage<ImageType> & myPointer, std::string filename )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typedef itk::ImageFileWriter< ImageType > ImageWriterType ;
    typename ImageWriterType::Pointer image_writer = ImageWriterType::New() ;
    image_writer->SetFileName( filename.c_str() ) ;
    image_writer->SetInput( image );
    image_writer->Update();
}


std::string ptrstr(void * c)
{
    std::stringstream ss;
    ss << (void const *)c;
    std::string s = ss.str();
    return s;
}


template <typename ImageType>
nb::object toNumpy( AntsImage<ImageType> & myPointer )
{
    typename ImageType::Pointer image = myPointer.ptr;
    typedef itk::PyBuffer<ImageType> PyBufferType;
    PyObject * itkArray = PyBufferType::_GetArrayViewFromImage( image );
    nb::object itkArrayObject = nb::steal( itkArray );
    return itkArrayObject;
}


void local_antsImage(nb::module_ &m) {
    m.def("ptrstr", &ptrstr);

    m.def("getShape",  &getShape<itk::Image<unsigned char,2>>);
    m.def("getShape",  &getShape<itk::Image<unsigned char,3>>);
    m.def("getShape",  &getShape<itk::Image<unsigned char,4>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,2>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,3>>);
    m.def("getShape",  &getShape<itk::Image<unsigned int,4>>);
    m.def("getShape",   &getShape<itk::Image<float,2>>);
    m.def("getShape",   &getShape<itk::Image<float,3>>);
    m.def("getShape",   &getShape<itk::Image<float,4>>);
    m.def("getShape",   &getShape<itk::Image<double,2>>);
    m.def("getShape",   &getShape<itk::Image<double,3>>);
    m.def("getShape",   &getShape<itk::Image<double,4>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,2>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,3>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned char,4>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,2>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,3>>);
    m.def("getShape", &getShape<itk::VectorImage<unsigned int,4>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,2>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,3>>);
    m.def("getShape",  &getShape<itk::VectorImage<float,4>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,2>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,3>>);
    m.def("getShape",  &getShape<itk::VectorImage<double,4>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getShape", &getShape<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,2>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,3>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned char,4>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,2>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,3>>);
    m.def("getOrigin",  &getOrigin<itk::Image<unsigned int,4>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,2>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,3>>);
    m.def("getOrigin",   &getOrigin<itk::Image<float,4>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,2>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,3>>);
    m.def("getOrigin",   &getOrigin<itk::Image<double,4>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("getOrigin", &getOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,2>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,3>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<float,4>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,2>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,3>>);
    m.def("getOrigin",  &getOrigin<itk::VectorImage<double,4>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getOrigin", &getOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,2>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,3>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned char,4>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,2>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,3>>);
    m.def("setOrigin",  &setOrigin<itk::Image<unsigned int,4>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,2>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,3>>);
    m.def("setOrigin",   &setOrigin<itk::Image<float,4>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,2>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,3>>);
    m.def("setOrigin",   &setOrigin<itk::Image<double,4>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,2>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,3>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned char,4>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,2>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,3>>);
    m.def("setOrigin", &setOrigin<itk::VectorImage<unsigned int,4>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,2>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,3>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<float,4>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,2>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,3>>);
    m.def("setOrigin",  &setOrigin<itk::VectorImage<double,4>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setOrigin", &setOrigin<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,2>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,3>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned char,4>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,2>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,3>>);
    m.def("getSpacing",  &getSpacing<itk::Image<unsigned int,4>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,2>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,3>>);
    m.def("getSpacing",   &getSpacing<itk::Image<float,4>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,2>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,3>>);
    m.def("getSpacing",   &getSpacing<itk::Image<double,4>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("getSpacing", &getSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,2>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,3>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<float,4>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,2>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,3>>);
    m.def("getSpacing",  &getSpacing<itk::VectorImage<double,4>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getSpacing", &getSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,2>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,3>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned char,4>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,2>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,3>>);
    m.def("setSpacing",  &setSpacing<itk::Image<unsigned int,4>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,2>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,3>>);
    m.def("setSpacing",   &setSpacing<itk::Image<float,4>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,2>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,3>>);
    m.def("setSpacing",   &setSpacing<itk::Image<double,4>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,2>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,3>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned char,4>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,2>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,3>>);
    m.def("setSpacing", &setSpacing<itk::VectorImage<unsigned int,4>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,2>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,3>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<float,4>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,2>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,3>>);
    m.def("setSpacing",  &setSpacing<itk::VectorImage<double,4>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setSpacing", &setSpacing<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("getDirection",  &getDirection<itk::Image<unsigned char,2>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned char,3>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned char,4>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,2>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,3>>);
    m.def("getDirection",  &getDirection<itk::Image<unsigned int,4>>);
    m.def("getDirection",   &getDirection<itk::Image<float,2>>);
    m.def("getDirection",   &getDirection<itk::Image<float,3>>);
    m.def("getDirection",   &getDirection<itk::Image<float,4>>);
    m.def("getDirection",   &getDirection<itk::Image<double,2>>);
    m.def("getDirection",   &getDirection<itk::Image<double,3>>);
    m.def("getDirection",   &getDirection<itk::Image<double,4>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,2>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,3>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned char,4>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,2>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,3>>);
    m.def("getDirection", &getDirection<itk::VectorImage<unsigned int,4>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,2>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,3>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<float,4>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,2>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,3>>);
    m.def("getDirection",  &getDirection<itk::VectorImage<double,4>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("getDirection", &getDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("setDirection",  &setDirection<itk::Image<unsigned char,2>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned char,3>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned char,4>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,2>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,3>>);
    m.def("setDirection",  &setDirection<itk::Image<unsigned int,4>>);
    m.def("setDirection",   &setDirection<itk::Image<float,2>>);
    m.def("setDirection",   &setDirection<itk::Image<float,3>>);
    m.def("setDirection",   &setDirection<itk::Image<float,4>>);
    m.def("setDirection",   &setDirection<itk::Image<double,2>>);
    m.def("setDirection",   &setDirection<itk::Image<double,3>>);
    m.def("setDirection",   &setDirection<itk::Image<double,4>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,2>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,3>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned char,4>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,2>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,3>>);
    m.def("setDirection", &setDirection<itk::VectorImage<unsigned int,4>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,2>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,3>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<float,4>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,2>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,3>>);
    m.def("setDirection",  &setDirection<itk::VectorImage<double,4>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("setDirection", &setDirection<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toFile",  &toFile<itk::Image<unsigned char,2>>);
    m.def("toFile",  &toFile<itk::Image<unsigned char,3>>);
    m.def("toFile",  &toFile<itk::Image<unsigned char,4>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,2>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,3>>);
    m.def("toFile",  &toFile<itk::Image<unsigned int,4>>);
    m.def("toFile",   &toFile<itk::Image<float,2>>);
    m.def("toFile",   &toFile<itk::Image<float,3>>);
    m.def("toFile",   &toFile<itk::Image<float,4>>);
    m.def("toFile",   &toFile<itk::Image<double,2>>);
    m.def("toFile",   &toFile<itk::Image<double,3>>);
    m.def("toFile",   &toFile<itk::Image<double,4>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,2>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,3>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned char,4>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,2>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,3>>);
    m.def("toFile", &toFile<itk::VectorImage<unsigned int,4>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,2>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,3>>);
    m.def("toFile",  &toFile<itk::VectorImage<float,4>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,2>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,3>>);
    m.def("toFile",  &toFile<itk::VectorImage<double,4>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toFile", &toFile<itk::Image<itk::RGBPixel<float>,3>>);

    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,2>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,3>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned char,4>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,2>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,3>>);
    m.def("toNumpy",  &toNumpy<itk::Image<unsigned int,4>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,2>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,3>>);
    m.def("toNumpy",   &toNumpy<itk::Image<float,4>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,2>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,3>>);
    m.def("toNumpy",   &toNumpy<itk::Image<double,4>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,2>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,3>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned char,4>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,2>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,3>>);
    m.def("toNumpy", &toNumpy<itk::VectorImage<unsigned int,4>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,2>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,3>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<float,4>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,2>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,3>>);
    m.def("toNumpy",  &toNumpy<itk::VectorImage<double,4>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,2>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<unsigned char>,3>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<float>,2>>);
    m.def("toNumpy", &toNumpy<itk::Image<itk::RGBPixel<float>,3>>);

    nb::class_<AntsImage<itk::Image<unsigned char,2>>>(m, "AntsImageUC2");
    nb::class_<AntsImage<itk::Image<unsigned char,3>>>(m, "AntsImageUC3");
    nb::class_<AntsImage<itk::Image<unsigned char,4>>>(m, "AntsImageUC4");
    nb::class_<AntsImage<itk::Image<unsigned int,2>>>(m, "AntsImageUI2");
    nb::class_<AntsImage<itk::Image<unsigned int,3>>>(m, "AntsImageUI3");
    nb::class_<AntsImage<itk::Image<unsigned int,4>>>(m, "AntsImageUI4");
    nb::class_<AntsImage<itk::Image<float,2>>>(m, "AntsImageF2");
    nb::class_<AntsImage<itk::Image<float,3>>>(m, "AntsImageF3");
    nb::class_<AntsImage<itk::Image<float,4>>>(m, "AntsImageF4");
    nb::class_<AntsImage<itk::Image<double,2>>>(m, "AntsImageD2");
    nb::class_<AntsImage<itk::Image<double,3>>>(m, "AntsImageD3");
    nb::class_<AntsImage<itk::Image<double,4>>>(m, "AntsImageD4");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,2>>>(m, "AntsImageVUC2");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,3>>>(m, "AntsImageVUC3");
    nb::class_<AntsImage<itk::VectorImage<unsigned char,4>>>(m, "AntsImageVUC4");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,2>>>(m, "AntsImageVUI2");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,3>>>(m, "AntsImageVUI3");
    nb::class_<AntsImage<itk::VectorImage<unsigned int,4>>>(m, "AntsImageVUI4");
    nb::class_<AntsImage<itk::VectorImage<float,2>>>(m, "AntsImageVF2");
    nb::class_<AntsImage<itk::VectorImage<float,3>>>(m, "AntsImageVF3");
    nb::class_<AntsImage<itk::VectorImage<float,4>>>(m, "AntsImageVF4");
    nb::class_<AntsImage<itk::VectorImage<double,2>>>(m, "AntsImageVD2");
    nb::class_<AntsImage<itk::VectorImage<double,3>>>(m, "AntsImageVD3");
    nb::class_<AntsImage<itk::VectorImage<double,4>>>(m, "AntsImageVD4");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<unsigned char>,2>>>(m, "AntsImageRGBUC2");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<unsigned char>,3>>>(m, "AntsImageRGBUC3");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<float>,2>>>(m, "AntsImageRGBF2");
    nb::class_<AntsImage<itk::Image<itk::RGBPixel<float>,3>>>(m, "AntsImageRGBF3");
}

