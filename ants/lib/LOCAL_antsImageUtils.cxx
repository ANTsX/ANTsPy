
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

#include "LOCAL_antsImage.h"

namespace py = pybind11;


template < typename ImageType >
std::vector<std::vector<float> > TransformIndexToPhysicalPoint( py::capsule antsImage,
                                                                std::vector<std::vector<int> > indices )
{
  typedef typename ImageType::Pointer                            ImagePointerType ;
  typedef typename ImageType::PointType                          PointType;
  typedef typename PointType::CoordRepType                       CoordRepType;

  typedef typename itk::ContinuousIndex<CoordRepType, ImageType::ImageDimension> IndexType;

  const unsigned int nDim = ImageType::ImageDimension;

  ImagePointerType image = as< ImageType >( antsImage );

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
std::vector<std::vector<float> > TransformPhysicalPointToIndex( py::capsule antsImage,
                                                                std::vector<std::vector<float> > points )
{
  typedef typename ImageType::Pointer      ImagePointerType ;
  typedef typename ImageType::PointType    PointType;
  typedef typename PointType::CoordRepType CoordRepType;

  typedef typename itk::ContinuousIndex<CoordRepType, ImageType::ImageDimension> IndexType;
  const unsigned int nDim = ImageType::ImageDimension;

  ImagePointerType image = as< ImageType >( antsImage );

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


PYBIND11_MODULE(antsImageUtils, m)
{
    m.def("TransformIndexToPhysicalPointUC2", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 2>>);
    m.def("TransformIndexToPhysicalPointUC3", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 3>>);
    m.def("TransformIndexToPhysicalPointUC4", &TransformIndexToPhysicalPoint<itk::Image<unsigned char, 4>>);
    m.def("TransformIndexToPhysicalPointUI2", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 2>>);
    m.def("TransformIndexToPhysicalPointUI3", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 3>>);
    m.def("TransformIndexToPhysicalPointUI4", &TransformIndexToPhysicalPoint<itk::Image<unsigned int, 4>>);
    m.def("TransformIndexToPhysicalPointF2", &TransformIndexToPhysicalPoint<itk::Image<float, 2>>);
    m.def("TransformIndexToPhysicalPointF3", &TransformIndexToPhysicalPoint<itk::Image<float, 3>>);
    m.def("TransformIndexToPhysicalPointF4", &TransformIndexToPhysicalPoint<itk::Image<float, 4>>);
    m.def("TransformIndexToPhysicalPointD2", &TransformIndexToPhysicalPoint<itk::Image<double, 2>>);
    m.def("TransformIndexToPhysicalPointD3", &TransformIndexToPhysicalPoint<itk::Image<double, 3>>);
    m.def("TransformIndexToPhysicalPointD4", &TransformIndexToPhysicalPoint<itk::Image<double, 4>>);

    m.def("TransformPhysicalPointToIndexUC2", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 2>>);
    m.def("TransformPhysicalPointToIndexUC3", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 3>>);
    m.def("TransformPhysicalPointToIndexUC4", &TransformPhysicalPointToIndex<itk::Image<unsigned char, 4>>);
    m.def("TransformPhysicalPointToIndexUI2", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 2>>);
    m.def("TransformPhysicalPointToIndexUI3", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 3>>);
    m.def("TransformPhysicalPointToIndexUI4", &TransformPhysicalPointToIndex<itk::Image<unsigned int, 4>>);
    m.def("TransformPhysicalPointToIndexF2", &TransformPhysicalPointToIndex<itk::Image<float, 2>>);
    m.def("TransformPhysicalPointToIndexF3", &TransformPhysicalPointToIndex<itk::Image<float, 3>>);
    m.def("TransformPhysicalPointToIndexF4", &TransformPhysicalPointToIndex<itk::Image<float, 4>>);
    m.def("TransformPhysicalPointToIndexD2", &TransformPhysicalPointToIndex<itk::Image<double, 2>>);
    m.def("TransformPhysicalPointToIndexD3", &TransformPhysicalPointToIndex<itk::Image<double, 3>>);
    m.def("TransformPhysicalPointToIndexD4", &TransformPhysicalPointToIndex<itk::Image<double, 4>>);

}
