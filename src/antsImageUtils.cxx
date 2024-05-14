
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>


#include <algorithm>
#include <vector>
#include <string>

#include "itkAddImageFilter.h"
#include "itkDefaultConvertPixelTraits.h"
#include "itkMultiplyImageFilter.h"
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageBase.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkNeighborhoodIterator.h"
#include "itkPermuteAxesImageFilter.h"
#include "itkCentralDifferenceImageFunction.h"
#include "itkContinuousIndex.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_determinant.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;


template < typename ImageType >
std::vector<std::vector<float> > TransformIndexToPhysicalPoint( AntsImage<ImageType> & antsImage,
                                                                std::vector<std::vector<int> > indices )
{
  typedef typename ImageType::Pointer                            ImagePointerType ;
  typedef typename ImageType::PointType                          PointType;
  typedef typename PointType::CoordRepType                       CoordRepType;

  typedef typename itk::ContinuousIndex<CoordRepType, ImageType::ImageDimension> IndexType;

  const unsigned int nDim = ImageType::ImageDimension;

  ImagePointerType image = antsImage.ptr;

  unsigned long N = indices.size(); // number of indices to convert
  //Rcpp::NumericMatrix points( N, nDim ) ;
  std::vector<std::vector<float> > points(N, std::vector<float>(nDim));

  IndexType itkindex;
  PointType itkpoint;

  for( unsigned int j = 0; j < N; j++)
    {

    for( unsigned int i = 0; i < nDim; i++ )
      {
      itkindex[i] = static_cast<CoordRepType>( indices[j][i] - 1.0 );
      }

    image->TransformContinuousIndexToPhysicalPoint( itkindex, itkpoint );

    for ( int i = 0; i < nDim; i++ )
      {
      points[j][i] = itkpoint[i];
      }
    }

  return points;
}


template < typename ImageType >
std::vector<std::vector<float> > TransformPhysicalPointToIndex( AntsImage<ImageType> & antsImage,
                                                                std::vector<std::vector<float> > points )
{
  typedef typename ImageType::Pointer      ImagePointerType ;
  typedef typename ImageType::PointType    PointType;
  typedef typename PointType::CoordRepType CoordRepType;

  typedef typename itk::ContinuousIndex<CoordRepType, ImageType::ImageDimension> IndexType;
  const unsigned int nDim = ImageType::ImageDimension;

  ImagePointerType image = antsImage.ptr;

  unsigned long N = points.size();
  std::vector<std::vector<float> > indices( N, std::vector<float>(nDim) );

  IndexType itkindex;
  PointType itkpoint;

  for( unsigned int j = 0; j < N; j++)
    {

    for( unsigned int i = 0; i < nDim; i++ )
      {
      itkpoint[i] = static_cast<CoordRepType>( points[j][i] );
      }

    image->TransformPhysicalPointToContinuousIndex( itkpoint, itkindex );

    for ( int i = 0; i < nDim; i++ )
      {
      indices[j][i] = itkindex[i] + 1.0;
      }
    }

  return indices;
}


void local_antsImageUtils(nb::module_ &m)
{
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 2>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 3>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 4>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 2>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 3>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 4>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<float, 2>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<float, 3>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<float, 4>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<double, 2>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<double, 3>>);
    m.def("TransformIndexToPhysicalPoint", &TransformIndexToPhysicalPoint<itk::Image<double, 4>>);

    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 2>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 3>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 4>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 2>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 3>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 4>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<float, 2>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<float, 3>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<float, 4>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<double, 2>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<double, 3>>);
    m.def("TransformPhysicalPointToIndex", &TransformPhysicalPointToIndex<itk::Image<double, 4>>);

}
