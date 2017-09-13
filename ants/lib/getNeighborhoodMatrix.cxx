
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

#include "antsImage.h"

namespace py = pybind11;
using namespace py::literals;

template< class PixelType , unsigned int Dimension >
py::dict getNeighborhoodMatrix( ANTsImage<itk::Image<PixelType,Dimension>> ants_image,
                                ANTsImage<itk::Image<PixelType,Dimension>> ants_mask,
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

  ImagePointerType image = as< ImageType >( ants_image );
  ImagePointerType mask = as< ImageType >( ants_mask );

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
                  matrix[row][col] = 0.0;
                  }
                }
              }
            else
              {
              matrix[row][col] = 0.0;
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
    return py::dict("matrix"_a=matrix );
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
                matrix[row][col] = 0.0;
                }
              }
            }
          else
            {
            matrix[row][col] = 0.0;
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
  return py::dict("values"_a=matrix, "gradients"_a=gradients);
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
              matrix[row][col] = 0.0;
              }
            }
          }
        else
          {
          matrix[row][col] = 0.0;
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
  return py::dict("values"_a=matrix, "indices"_a=indices, "offsets"_a=offsets);
}

PYBIND11_MODULE(getNeighborhoodMatrix, m)
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
    
    m.def("getNeighborhoodMatrixD2", &getNeighborhoodMatrix<double,2>);
    m.def("getNeighborhoodMatrixD3", &getNeighborhoodMatrix<double,3>);
    m.def("getNeighborhoodMatrixD4", &getNeighborhoodMatrix<double,4>);
}



