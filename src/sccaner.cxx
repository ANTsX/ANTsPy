
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <exception>
#include <vector>
#include <string>
#include "itkImage.h"
#include "itkVectorImage.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "antscore/antsSCCANObject.h"
#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template< class ImageType, class IntType, class RealType >
nb::dict sccanCppHelper(
  std::vector<std::vector<double>> X,
  std::vector<std::vector<double>> Y,
  AntsImage<ImageType> & maskXimage,
  AntsImage<ImageType> & maskYimage,
  int maskxisnull,
  int maskyisnull,
  RealType sparsenessx,
  RealType sparsenessy,
  IntType nvecs,
  IntType its,
  IntType cthreshx,
  IntType cthreshy,
  RealType z,
  RealType smooth,
  std::vector<AntsImage<ImageType>> initializationListx,
  std::vector<AntsImage<ImageType>> initializationListy,
  IntType covering,
  RealType ell1,
  IntType verbose,
  RealType priorWeight,
  IntType useMaxBasedThresh )
{
  enum { Dimension = ImageType::ImageDimension };
  typename ImageType::RegionType region;
  typedef typename ImageType::PixelType PixelType;
  typedef typename ImageType::Pointer ImagePointerType;
  typedef double                                        Scalar;
  typedef itk::ants::antsSCCANObject<ImageType, Scalar> SCCANType;
  typedef typename SCCANType::MatrixType                vMatrix;
  typename SCCANType::Pointer sccanobj = SCCANType::New();
  sccanobj->SetMaxBasedThresholding( useMaxBasedThresh );

  // cast mask ANTsImages to itk
  typename ImageType::Pointer maskx = ITK_NULLPTR;
  if (maskxisnull > 0)
  {
    maskx = maskXimage.ptr;
  }
  typename ImageType::Pointer masky = ITK_NULLPTR;
  if (maskyisnull > 0)
  {
    masky = maskYimage.ptr;
  }

// deal with the initializationList, if any
  unsigned int nImagesx = initializationListx.size();
  if ( ( nImagesx > 0 ) && ( !maskxisnull ) )
  {
    itk::ImageRegionIteratorWithIndex<ImageType> it( maskx,
      maskx->GetLargestPossibleRegion() );
    vMatrix priorROIMatx( nImagesx , X[0].size() );
    priorROIMatx.fill( 0 );
    for ( unsigned int i = 0; i < nImagesx; i++ )
    {
      typename ImageType::Pointer init = initializationListx[i].ptr;
      unsigned long ct = 0;
      it.GoToBegin();
      while ( !it.IsAtEnd() )
      {
        PixelType pix = it.Get();
        if ( pix >= 0.5 )
        {
          pix = init->GetPixel( it.GetIndex() );
          priorROIMatx( i, ct ) = pix;
          ct++;
        }
        ++it;
      }
    }
    sccanobj->SetMatrixPriorROI( priorROIMatx );
    nvecs = nImagesx;
  }
  unsigned int nImagesy = initializationListy.size();
  if ( ( nImagesy > 0 ) && ( !maskyisnull ) )
  {
    itk::ImageRegionIteratorWithIndex<ImageType> it( masky,
      masky->GetLargestPossibleRegion() );
    vMatrix priorROIMaty( nImagesy , Y[0].size() );
    priorROIMaty.fill( 0 );
    for ( unsigned int i = 0; i < nImagesy; i++ )
    {
      typename ImageType::Pointer init = initializationListy[i].ptr;
      unsigned long ct = 0;
      it.GoToBegin();
      while ( !it.IsAtEnd() )
      {
        PixelType pix = it.Get();
        if ( pix >= 0.5 )
        {
          pix = init->GetPixel( it.GetIndex() );
          priorROIMaty( i, ct ) = pix;
          ct++;
        }
        ++it;
      }
    }
    sccanobj->SetMatrixPriorROI2( priorROIMaty );
    nvecs = nImagesy;
  }
  sccanobj->SetPriorWeight( priorWeight );
  sccanobj->SetLambda( priorWeight );
// cast hack from Python type to sccan type
  //std::vector<double> xdat = X;//.reshape({-1}).cast<std::vector<double> >();
  //const double* _xdata = &xdat[0];

    vMatrix vnlX(  X.size(), X[0].size() );
  for (int i = 0; i < X.size(); i++)
  {
    for (int j = 0; j < X[0].size(); j++)
    {
      vnlX(i,j) = X[i][j];
    }
  }

  //vnlX = vnlX.transpose();

  //std::vector<double> ydat = Y.reshape({-1}).cast<std::vector<double> >();
  //const double* _ydata = &ydat[0];
  //vMatrix vnlY( _ydata , Y.shape(0), Y.shape(1)  );
    vMatrix vnlY(  Y.size(), Y[0].size() );
  for (int i = 0; i < Y.size(); i++)
  {
    for (int j = 0; j < Y[0].size(); j++)
    {
      vnlY(i,j) = Y[i][j];
    }
  }
  //vnlY = vnlY.transpose();
// cast hack done
  sccanobj->SetGetSmall( false  );
  sccanobj->SetCovering( covering );
  sccanobj->SetSilent(  ! verbose  );
  if( ell1 > 0 )
    {
    sccanobj->SetUseL1( true );
    }
  else
    {
    sccanobj->SetUseL1( false );
    }
  sccanobj->SetGradStep( std::abs( ell1 ) );
  sccanobj->SetMaximumNumberOfIterations( its );
  sccanobj->SetRowSparseness( z );
  sccanobj->SetSmoother( smooth );
  if ( sparsenessx < 0 ) sccanobj->SetKeepPositiveP(false);
  if ( sparsenessy < 0 ) sccanobj->SetKeepPositiveQ(false);
  sccanobj->SetSCCANFormulation(  SCCANType::PQ );
  sccanobj->SetFractionNonZeroP( fabs( sparsenessx ) );
  sccanobj->SetFractionNonZeroQ( fabs( sparsenessy ) );
  sccanobj->SetMinClusterSizeP( cthreshx );
  sccanobj->SetMinClusterSizeQ( cthreshy );
  sccanobj->SetMatrixP( vnlX );
  sccanobj->SetMatrixQ( vnlY );
//  sccanobj->SetMatrixR( r ); // FIXME
  sccanobj->SetMaskImageP( maskx );
  sccanobj->SetMaskImageQ( masky );
  sccanobj->SparsePartialArnoldiCCA( nvecs );

  // FIXME - should not copy, should map memory
  vMatrix solP = sccanobj->GetVariatesP();
  std::vector<std::vector<float> > eanatMatp( solP.cols(), std::vector<float>(solP.rows()));
  unsigned long rows = solP.rows();
  for( unsigned long c = 0; c < solP.cols(); c++ )
    {
    for( unsigned int r = 0; r < rows; r++ )
      {
      eanatMatp[c][r] = solP( r, c );
      }
    }

  vMatrix solQ = sccanobj->GetVariatesQ();

  std::vector<std::vector<float> > eanatMatq( solQ.cols(), std::vector<float>(solQ.rows()));
  rows = solQ.rows();
  for( unsigned long c = 0; c < solQ.cols(); c++ )
    {
    for( unsigned int r = 0; r < rows; r++ )
      {
      eanatMatq[c][r] = solQ( r, c );
      }
    }

  //nb::ndarray<double> eanatMatpList = nb::cast(eanatMatp);
  //nb::ndarray<double> eanatMatqList = nb::cast(eanatMatq);

     nb::dict res;
   res["eig1"] = eanatMatp; 
   res["eig2"] = eanatMatq;
   return res;

}

template <typename ImageType>
nb::dict sccanCpp(std::vector<std::vector<double>> X,
                   std::vector<std::vector<double>> Y,
                   AntsImage<ImageType> & maskXimage,
                   AntsImage<ImageType> & maskYimage,
                   int maskxisnull,
                   int maskyisnull,
                   float sparsenessx,
                   float sparsenessy,
                   unsigned int nvecs,
                   unsigned int its,
                   unsigned int cthreshx,
                   unsigned int cthreshy,
                   float z,
                   float smooth,
                   std::vector<AntsImage<ImageType>> initializationListx,
                   std::vector<AntsImage<ImageType>> initializationListy,
                   float ell1,
                   unsigned int verbose,
                   float priorWeight,
                   unsigned int mycoption,
                   unsigned int maxBasedThresh)
{
  typedef float RealType;
  typedef unsigned int IntType;
  return  sccanCppHelper<ImageType,IntType,RealType>(
        X,
        Y,
        maskXimage,
        maskYimage,
        maskxisnull,
        maskyisnull,
        sparsenessx,
        sparsenessy,
        nvecs,
        its,
        cthreshx,
        cthreshy,
        z,
        smooth,
        initializationListx,
        initializationListy,
        mycoption,
        ell1,
        verbose,
        priorWeight,
        maxBasedThresh
        );

}


template< class ImageType, class IntType, class RealType >
nb::dict sccanCppHelperV2(
  std::vector<std::vector<double> > X,
  std::vector<std::vector<double> > Y,
  AntsImage<ImageType> & maskXimage,
  AntsImage<ImageType> & maskYimage,
  int maskxisnull,
  int maskyisnull,
  RealType sparsenessx,
  RealType sparsenessy,
  IntType nvecs,
  IntType its,
  IntType cthreshx,
  IntType cthreshy,
  RealType z,
  RealType smooth,
  std::vector<AntsImage<ImageType>> initializationListx,
  std::vector<AntsImage<ImageType>> initializationListy,
  IntType covering,
  RealType ell1,
  IntType verbose,
  RealType priorWeight,
  IntType useMaxBasedThresh )
{
  enum { Dimension = ImageType::ImageDimension };
  typename ImageType::RegionType region;
  typedef typename ImageType::PixelType PixelType;
  typedef typename ImageType::Pointer ImagePointerType;
  typedef double                                        Scalar;
  typedef itk::ants::antsSCCANObject<ImageType, Scalar> SCCANType;
  typedef typename SCCANType::MatrixType                vMatrix;
  typename SCCANType::Pointer sccanobj = SCCANType::New();
  sccanobj->SetMaxBasedThresholding( useMaxBasedThresh );

  // cast mask ANTsImages to itk
  typename ImageType::Pointer maskx = ITK_NULLPTR;
  if (maskxisnull > 0)
  {
    maskx = maskXimage.ptr;
  }
  typename ImageType::Pointer masky = ITK_NULLPTR;
  if (maskyisnull > 0)
  {
    masky = maskYimage.ptr;
  }

// deal with the initializationList, if any
  unsigned int nImagesx = initializationListx.size();
  if ( ( nImagesx > 0 ) && ( !maskxisnull ) )
  {
    itk::ImageRegionIteratorWithIndex<ImageType> it( maskx,
      maskx->GetLargestPossibleRegion() );
    vMatrix priorROIMatx( nImagesx , X[0].size() );
    priorROIMatx.fill( 0 );
    for ( unsigned int i = 0; i < nImagesx; i++ )
    {
      typename ImageType::Pointer init = initializationListx[i].ptr;
      unsigned long ct = 0;
      it.GoToBegin();
      while ( !it.IsAtEnd() )
      {
        PixelType pix = it.Get();
        if ( pix >= 0.5 )
        {
          pix = init->GetPixel( it.GetIndex() );
          priorROIMatx( i, ct ) = pix;
          ct++;
        }
        ++it;
      }
    }
    sccanobj->SetMatrixPriorROI( priorROIMatx );
    nvecs = nImagesx;
  }
  unsigned int nImagesy = initializationListy.size();
  if ( ( nImagesy > 0 ) && ( !maskyisnull ) )
  {
    itk::ImageRegionIteratorWithIndex<ImageType> it( masky,
      masky->GetLargestPossibleRegion() );
    vMatrix priorROIMaty( nImagesy , Y[0].size() );
    priorROIMaty.fill( 0 );
    for ( unsigned int i = 0; i < nImagesy; i++ )
    {
      typename ImageType::Pointer init = initializationListy[i].ptr;
      unsigned long ct = 0;
      it.GoToBegin();
      while ( !it.IsAtEnd() )
      {
        PixelType pix = it.Get();
        if ( pix >= 0.5 )
        {
          pix = init->GetPixel( it.GetIndex() );
          priorROIMaty( i, ct ) = pix;
          ct++;
        }
        ++it;
      }
    }
    sccanobj->SetMatrixPriorROI2( priorROIMaty );
    nvecs = nImagesy;
  }
  sccanobj->SetPriorWeight( priorWeight );
  sccanobj->SetLambda( priorWeight );
// cast hack from Python type to sccan type
  //std::vector<std::vector<double> > xdat;
  //const double* _xdata = &X[0][0];
  vMatrix vnlX( X.size(), X[0].size()  );
  for (int i = 0; i < X.size(); i++)
  {
    for (int j = 0; j < X[0].size(); j++)
    {
      vnlX(i,j) = X[i][j];
    }
  }
  //vnlX = vnlX.transpose();

  //std::vector<std::vector<double> > ydat;
  //const double*  _ydata = &Y[0][0];
  vMatrix vnlY(  Y.size(), Y[0].size() );
  for (int i = 0; i < Y.size(); i++)
  {
    for (int j = 0; j < Y[0].size(); j++)
    {
      vnlY(i,j) = Y[i][j];
    }
  }
// cast hack done
  sccanobj->SetGetSmall( false  );
  sccanobj->SetCovering( covering );
  sccanobj->SetSilent(  ! verbose  );
  if( ell1 > 0 )
    {
    sccanobj->SetUseL1( true );
    }
  else
    {
    sccanobj->SetUseL1( false );
    }
  sccanobj->SetGradStep( std::abs( ell1 ) );
  sccanobj->SetMaximumNumberOfIterations( its );
  sccanobj->SetRowSparseness( z );
  sccanobj->SetSmoother( smooth );
  if ( sparsenessx < 0 ) sccanobj->SetKeepPositiveP(false);
  if ( sparsenessy < 0 ) sccanobj->SetKeepPositiveQ(false);
  sccanobj->SetSCCANFormulation(  SCCANType::PQ );
  sccanobj->SetFractionNonZeroP( fabs( sparsenessx ) );
  sccanobj->SetFractionNonZeroQ( fabs( sparsenessy ) );
  sccanobj->SetMinClusterSizeP( cthreshx );
  sccanobj->SetMinClusterSizeQ( cthreshy );
  sccanobj->SetMatrixP( vnlX );
  sccanobj->SetMatrixQ( vnlY );
//  sccanobj->SetMatrixR( r ); // FIXME
  sccanobj->SetMaskImageP( maskx );
  sccanobj->SetMaskImageQ( masky );
  sccanobj->SparsePartialArnoldiCCA( nvecs );

  // FIXME - should not copy, should map memory
  vMatrix solP = sccanobj->GetVariatesP();
  std::vector<std::vector<double> > eanatMatp( solP.cols(), std::vector<double>(solP.rows()));
  unsigned long rows = solP.rows();
  for( unsigned long c = 0; c < solP.cols(); c++ )
    {
    for( unsigned int r = 0; r < rows; r++ )
      {
      eanatMatp[c][r] = solP( r, c );
      }
    }

  vMatrix solQ = sccanobj->GetVariatesQ();

  std::vector<std::vector<double> > eanatMatq( solQ.cols(), std::vector<double>(solQ.rows()));
  rows = solQ.rows();
  for( unsigned long c = 0; c < solQ.cols(); c++ )
    {
    for( unsigned int r = 0; r < rows; r++ )
      {
      eanatMatq[c][r] = solQ( r, c );
      }
    }

  //nb::ndarray<double> eanatMatpList = nb::cast(eanatMatp);
  //nb::ndarray<double> eanatMatqList = nb::cast(eanatMatq);

   nb::dict res;
   res["eig1"] = eanatMatp; 
   res["eig2"] = eanatMatq;
   return res;

}

template <typename ImageType>
nb::dict sccanCppV2(std::vector<std::vector<double> > X,
                   std::vector<std::vector<double> >  Y,
                   AntsImage<ImageType> & maskXimage,
                   AntsImage<ImageType> & maskYimage,
                   int maskxisnull,
                   int maskyisnull,
                   float sparsenessx,
                   float sparsenessy,
                   unsigned int nvecs,
                   unsigned int its,
                   unsigned int cthreshx,
                   unsigned int cthreshy,
                   float z,
                   float smooth,
                   std::vector<AntsImage<ImageType>> initializationListx,
                   std::vector<AntsImage<ImageType>> initializationListy,
                   float ell1,
                   unsigned int verbose,
                   float priorWeight,
                   unsigned int mycoption,
                   unsigned int maxBasedThresh)
{
  typedef float RealType;
  typedef unsigned int IntType;
  return  sccanCppHelperV2<ImageType,IntType,RealType>(
        X,
        Y,
        maskXimage,
        maskYimage,
        maskxisnull,
        maskyisnull,
        sparsenessx,
        sparsenessy,
        nvecs,
        its,
        cthreshx,
        cthreshy,
        z,
        smooth,
        initializationListx,
        initializationListy,
        mycoption,
        ell1,
        verbose,
        priorWeight,
        maxBasedThresh
        );

}

void local_sccaner(nb::module_ &m)
{
  m.def("sccanCpp2D", &sccanCpp<itk::Image<float, 2>>);
  m.def("sccanCpp3D", &sccanCpp<itk::Image<float, 3>>);
  m.def("sccanCpp2DV2", &sccanCppV2<itk::Image<float, 2>>);
  m.def("sccanCpp3DV2", &sccanCppV2<itk::Image<float, 3>>);
}
