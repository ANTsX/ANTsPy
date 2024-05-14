
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
#include <limits>

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

template< class PixelType , unsigned int Dimension >
nb::dict getNeighborhoodMatrix( AntsImage<itk::Image<PixelType, Dimension>> & ants_image,
                                AntsImage<itk::Image<PixelType, Dimension>> & ants_mask,
                                std::vector<int> radius,
                                int physical,
                                int boundary,
                                int spatial,
                                int getgradient)
{

  typedef double                           RealType;
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::Pointer      ImagePointerType;
  typedef typename ImageType::IndexType    IndexType;
  typedef typename ImageType::PointType    PointType;
  typedef itk::CentralDifferenceImageFunction< ImageType, RealType >  GradientCalculatorType;
  typedef itk::CovariantVector<RealType, Dimension> CovariantVectorType;

  ImagePointerType image = ants_image.ptr;
  ImagePointerType mask = ants_mask.ptr;

  //Rcpp::NumericVector radius( r_radius ) ;
  //int physical = Rcpp::as<int>( r_physical );
  //int boundary = Rcpp::as<int>( r_boundary );
  //int spatial = Rcpp::as<int>( r_spatial );
  //int getgradient = Rcpp::as<int>( r_gradient );

  typename itk::NeighborhoodIterator<ImageType>::SizeType nSize;

  unsigned long maxSize = 1;
  for ( unsigned int i=0; i<Dimension; i++ )
    {
    maxSize *= ( 1 + 2*radius[i] );
    nSize[i] = radius[i];
    }

  std::vector<double> pixelList;
  pixelList.reserve(maxSize);

  itk::ImageRegionIteratorWithIndex<ImageType> it( mask, mask->GetLargestPossibleRegion() ) ;
  itk::NeighborhoodIterator<ImageType> nit( nSize, image, image->GetLargestPossibleRegion() ) ;

  unsigned long nVoxels = 0;
  while( !it.IsAtEnd() )
    {
    if ( it.Value() > 0 )
      {
      ++nVoxels;
      }
    ++it;
    }

  //Rcpp::NumericMatrix matrix(maxSize, nVoxels);
    std::vector<std::vector<float> > matrix(maxSize, std::vector<float>(nVoxels));
  if ( ( ! spatial )  && ( ! getgradient ) )
    {
      unsigned int col = 0;
      it.GoToBegin();
      while( !it.IsAtEnd() )
        {
        if ( it.Value() > 1.e-6 ) // use epsilon instead of zero
          {
          double mean = 0;
          double count = 0;
          for ( unsigned int row=0; row < nit.Size(); row++ )
            {
            IndexType idx = it.GetIndex() + nit.GetOffset(row);

            // check boundary conditions
            if ( mask->GetRequestedRegion().IsInside(idx) )
              {
              if ( mask->GetPixel(idx) > 0 ) // fully within boundaries
                {
                matrix[row][col] = nit.GetPixel(row);
                mean += nit.GetPixel(row);
                ++count;
                }
              else
                {
                if ( boundary == 1 )
                  {
                  matrix[row][col]  = nit.GetPixel(row);
                  }
                else
                  {
                  matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
                  }
                }
              }
            else
              {
              matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
              }
            }

          if ( boundary == 2 )
            {
            mean /= count;
            for ( unsigned int row=0; row < nit.Size(); row++ )
              {
              if ( matrix[row][col] != matrix[row][col] )
                {
                matrix[row][col] = mean;
                }
              }
            }

          ++col;
          }
        ++it;
        ++nit;
        }

    nb::dict mymat;
    mymat["matrix"] = matrix;
    return mymat;
    }

  if ( ( ! spatial )  && ( getgradient ) )
  {
    typename GradientCalculatorType::Pointer
      imageGradientCalculator = GradientCalculatorType::New();

    imageGradientCalculator->SetInputImage( image );
    // this will hold spatial locations of pixels or voxels
    //Rcpp::NumericMatrix gradients( Dimension, nVoxels );
    std::vector<std::vector<float> > gradients(Dimension, std::vector<float>(nVoxels));
    unsigned int col = 0;
    it.GoToBegin();
    while( !it.IsAtEnd() )
      {
      if ( it.Value() > 1.e-6 ) // use epsilon instead of zero
        {
        double mean = 0;
        double count = 0;
        for ( unsigned int row=0; row < nit.Size(); row++ )
          {
          IndexType idx = it.GetIndex() + nit.GetOffset(row);

          // check boundary conditions
          if ( mask->GetRequestedRegion().IsInside(idx) )
            {
            if ( mask->GetPixel(idx) > 0 ) // fully within boundaries
              {
              matrix[row][col] = nit.GetPixel(row);
              mean += nit.GetPixel(row);
              ++count;
              if ( row == 0 )
                {
                CovariantVectorType gradient =
                  imageGradientCalculator->EvaluateAtIndex( it.GetIndex() );
                for ( unsigned int dd = 0; dd < Dimension; dd++ )
                  gradients[dd][col] = gradient[ dd ];
                }
              }
            else
              {
              if ( boundary == 1 )
                {
                matrix[row][col] = nit.GetPixel(row);
                }
              else
                {
                matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
                }
              }
            }
          else
            {
            matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
            }
          }

        if ( boundary == 2 )
          {
          mean /= count;
          for ( unsigned int row=0; row < nit.Size(); row++ )
            {
            if ( matrix[row][col] != matrix[row][col] )
              {
              matrix[row][col] = mean;
              }
            }
          }

        ++col;
        }
      ++it;
      ++nit;
      }
  //return Rcpp::List::create( Rcpp::Named("values") = matrix,
  //                           Rcpp::Named("gradients") = gradients );
    nb::dict res;
    res["values"] = matrix;
    res["gradients"] = gradients;
  return res;
  }
// if spatial and gradient, then just use spatial - no gradient ...

  // this will hold spatial locations of pixels or voxels
  //Rcpp::NumericMatrix indices(nVoxels, Dimension);
  std::vector<std::vector<float> > indices(nVoxels, std::vector<float>(Dimension));

  // Get relative offsets of neighborhood locations
  //Rcpp::NumericMatrix offsets(nit.Size(), Dimension);
  std::vector<std::vector<float> > offsets(nit.Size(), std::vector<float>(Dimension));

  for ( unsigned int i=0; i < nit.Size(); i++ )
    {
    for ( unsigned int j=0; j<Dimension; j++)
      {
      offsets[i][j] = nit.GetOffset(i)[j];
      if ( physical )
        {
        offsets[i][j] = offsets[i][j] * image->GetSpacing()[j];
        }
      }
    }


  unsigned int col = 0;
  it.GoToBegin();
  while( !it.IsAtEnd() )
    {
    if ( it.Value() > 0 )
      {
      PointType pt;

      if ( physical )
        {
        image->TransformIndexToPhysicalPoint(it.GetIndex(), pt);
        }

      for ( unsigned int i=0; i < Dimension; i++)
        {
        if ( physical )
          {
          indices[col][i] = pt[i];
          }
        else
          {
          indices[col][i] = it.GetIndex()[i] + 1;
          }
        }

      double mean = 0;
      double count = 0;
      for ( unsigned int row=0; row < nit.Size(); row++ )
        {
        IndexType idx = it.GetIndex() + nit.GetOffset(row);

        // check boundary conditions
        if ( mask->GetRequestedRegion().IsInside(idx) )
          {
          if ( mask->GetPixel(idx) > 0 ) // fully within boundaries
            {
            matrix[row][col] = nit.GetPixel(row);
            mean += nit.GetPixel(row);
            ++count;
            }
          else
            {
            if ( boundary == 1 )
              {
              matrix[row][col]  = nit.GetPixel(row);
              }
            else
              {
              matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
              }
            }
          }
        else
          {
          matrix[row][col] = std::numeric_limits<double>::quiet_NaN();
          }
        }

      if ( boundary == 2 )
        {
        mean /= count;
        for ( unsigned int row=0; row < nit.Size(); row++ )
          {
          if ( matrix[row][col] != matrix[row][col] )
            {
            matrix[row][col] = mean;
            }
          }
        }

      ++col;
      }
    ++it;
    ++nit;
    }
  //return Rcpp::List::create( Rcpp::Named("values") = matrix,
  //                             Rcpp::Named("indices") = indices,
  //                             Rcpp::Named("offsets") = offsets );
    nb::dict res;
    res["values"] = matrix;
    res["indices"] = indices;
    res["offsets"] = offsets;
  return res;
}


template< class PixelType , unsigned int Dimension >
nb::dict getNeighborhood( AntsImage<itk::Image<PixelType, Dimension>> & ants_image,
                          std::vector<float> index,
                          std::vector<float> kernel,
                          std::vector<int> radius,
                          int physicalFlag )
{

  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef typename ImageType::Pointer      ImagePointerType;
  typedef typename ImageType::RegionType   RegionType;
  typedef typename ImageType::IndexType    IndexType;

  ImagePointerType image = ants_image.ptr;

  //Rcpp::NumericVector kernel( r_kernel );
  //Rcpp::NumericVector radius( r_radius );
  //Rcpp::NumericVector index( r_index );
  //int physicalFlag = Rcpp::as<int>( physical );

  unsigned long maxSize = 0;
  std::vector<int> offsets;
  for ( unsigned int i=0; i<kernel.size(); i++ )
    {
    if ( kernel[i] > 1.e-6 ) // use epsilon instead of zero
      {
      offsets.push_back(i);
      ++maxSize;
      }
    }

  //Rcpp::NumericVector pixels(maxSize);
  std::vector<float> pixels(maxSize);
  std::vector<IndexType> indexList;
  indexList.reserve(maxSize);

  RegionType region;
  typename itk::NeighborhoodIterator<ImageType>::SizeType nRadius;

  for ( unsigned int i=0; i<Dimension; i++ )
    {
    nRadius[i] = radius[i];
    region.SetSize(i, 1);
    region.SetIndex(i, index[i]-1); // R-to-ITK index conversion
    }

  RegionType imageSize = image->GetLargestPossibleRegion();
  itk::NeighborhoodIterator<ImageType> nit( nRadius, image, region );

  unsigned int idx = 0;
  for (unsigned int i = 0; i < offsets.size(); i++ )
    {
    //Rcpp::Rcout << nit.GetIndex(i) << ":" << offsets[i] << "=" << kernel[offsets[i]] << std::endl;
    if ( kernel[offsets[i]] > 1e-6 )
      {
      if ( imageSize.IsInside( nit.GetIndex(offsets[i]) ) )
        {
        pixels[idx] = nit.GetPixel(offsets[i]);
        }
      else
        {
        pixels[idx] = 0.0;
        }
      indexList.push_back( nit.GetIndex(offsets[i]) );
      ++idx;
      }
    }

  //Rcpp::NumericMatrix indices( pixels.size(), Dimension );
  std::vector<std::vector<float> > indices(pixels.size(), std::vector<float>(Dimension));

  for ( unsigned int i=0; i<pixels.size(); i++)
    {
    typename ImageType::PointType pt;
    if ( physicalFlag )
      {
      image->TransformIndexToPhysicalPoint( indexList[i], pt );
      }

    for ( unsigned int j=0; j<Dimension; j++)
      {
      if ( !physicalFlag )
        {
        indices[i][j] = indexList[i][j] + 1; // ITK-to-R index conversion
        }
      else
        {
        indices[i][j] = pt[j];
        }
      }
    }

  //return Rcpp::List::create( Rcpp::Named("values") = pixels,
  //                           Rcpp::Named("indices") = indices );

    nb::dict res;
    res["values"] = pixels;
    res["indices"] = indices;
  return res;

}


void local_getNeighborhoodMatrix(nb::module_ &m)
{
    m.def("getNeighborhoodMatrixUC2", &getNeighborhoodMatrix<unsigned char,2>);
    m.def("getNeighborhoodMatrixUC3", &getNeighborhoodMatrix<unsigned char,3>);
    m.def("getNeighborhoodMatrixUC4", &getNeighborhoodMatrix<unsigned char,4>);

    m.def("getNeighborhoodMatrixUI2", &getNeighborhoodMatrix<unsigned int,2>);
    m.def("getNeighborhoodMatrixUI3", &getNeighborhoodMatrix<unsigned int,3>);
    m.def("getNeighborhoodMatrixUI4", &getNeighborhoodMatrix<unsigned int,4>);

    m.def("getNeighborhoodMatrixF2", &getNeighborhoodMatrix<float,2>);
    m.def("getNeighborhoodMatrixF3", &getNeighborhoodMatrix<float,3>);
    m.def("getNeighborhoodMatrixF4", &getNeighborhoodMatrix<float,4>);

    m.def("getNeighborhoodUC2", &getNeighborhood<unsigned char,2>);
    m.def("getNeighborhoodUC3", &getNeighborhood<unsigned char,3>);
    m.def("getNeighborhoodUC4", &getNeighborhood<unsigned char,4>);

    m.def("getNeighborhoodUI2", &getNeighborhood<unsigned int,2>);
    m.def("getNeighborhoodUI3", &getNeighborhood<unsigned int,3>);
    m.def("getNeighborhoodUI4", &getNeighborhood<unsigned int,4>);

    m.def("getNeighborhoodF2", &getNeighborhood<float,2>);
    m.def("getNeighborhoodF3", &getNeighborhood<float,3>);
    m.def("getNeighborhoodF4", &getNeighborhood<float,4>);
}



