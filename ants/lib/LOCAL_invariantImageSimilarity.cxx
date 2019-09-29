
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include <algorithm>
#include <vnl/vnl_inverse.h>
#include <vnl/vnl_cross.h>
#include "itkImageMaskSpatialObject.h"
#include "itkANTSNeighborhoodCorrelationImageToImageMetricv4.h"
#include "itkArray.h"
#include "itkGradientImageFilter.h"
#include "itkBSplineControlPointImageFilter.h"
#include "itkBayesianClassifierImageFilter.h"
#include "itkBayesianClassifierInitializationImageFilter.h"
#include "itkBilateralImageFilter.h"
#include "itkCSVNumericObjectFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkCompositeValleyFunction.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkConvolutionImageFilter.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkDistanceToCentroidMembershipFunction.h"
#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkDemonsImageToImageMetricv4.h"
#include "itkExpImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkGaussianImageSource.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkHistogram.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkImage.h"
#include "itkImageClassifierBase.h"
#include "itkImageDuplicator.h"
#include "itkImageFileWriter.h"
#include "itkImageGaussianModelEstimator.h"
#include "itkImageKmeansModelEstimator.h"
#include "itkImageMomentsCalculator.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkKdTree.h"
#include "itkKdTreeBasedKmeansEstimator.h"
#include "itkLabelContourImageFilter.h"
#include "itkLabelStatisticsImageFilter.h"
#include "itkLaplacianRecursiveGaussianImageFilter.h"
#include "itkListSample.h"
#include "itkMRFImageFilter.h"
#include "itkMRIBiasFieldCorrectionFilter.h"
#include "itkMaskImageFilter.h"
#include "itkMaximumImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkMultivariateLegendrePolynomial.h"
#include "itkMultiStartOptimizerv4.h"
#include "itkNeighborhood.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkNeighborhoodIterator.h"
#include "itkNormalVariateGenerator.h"
#include "itkOptimizerParameterScalesEstimator.h"
#include "itkOtsuThresholdImageFilter.h"
#include "itkRGBPixel.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSampleToHistogramFilter.h"
#include "itkScalarImageKmeansImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkSimilarity3DTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkSize.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkTranslationTransform.h"
#include "itkVariableSizeMatrix.h"
#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "vnl/vnl_matrix_fixed.h"
#include "itkTransformFactory.h"
#include "itkEuler2DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkCenteredAffineTransform.h"
#include "itkCompositeTransform.h"

#include <fstream>
#include <iostream>
#include <map> // Here I'm using a map but you could choose even other containers
#include <sstream>
#include <string>

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <class TComputeType, unsigned int ImageDimension>
class SimilarityTransformTraits
{
  // Don't worry about the fact that the default option is the
  // affine Transform, that one will not actually be instantiated.
public:
  typedef itk::AffineTransform<TComputeType, ImageDimension> TransformType;
};


template <>
class SimilarityTransformTraits<double, 2>
{
public:
  typedef itk::Similarity2DTransform<double> TransformType;
};

template <>
class SimilarityTransformTraits<float, 2>
{
public:
  typedef itk::Similarity2DTransform<float> TransformType;
};

template <>
class SimilarityTransformTraits<double, 3>
{
public:
  typedef itk::Similarity3DTransform<double> TransformType;
};

template <>
class SimilarityTransformTraits<float, 3>
{
public:
  typedef itk::Similarity3DTransform<float> TransformType;
};


template <typename AffineType, unsigned int ImageDimension >
std::vector<std::vector<float> > invariantImageSimilarity(py::capsule in_image1,
                                                          py::capsule in_image2,
                                                          std::vector<float> thetas,
                                                          std::vector<float> thetas2,
                                                          std::vector<float> thetas3,
                                                          unsigned int localSearchIterations,
                                                          std::string whichMetric,
                                                          float bestscale,
                                                          unsigned int doReflection,
                                                          std::string txfn  )
{
  typedef float  PixelType;
  typedef double RealType;
  typedef itk::Image< PixelType , ImageDimension > ImageType;
  typedef typename ImageType::Pointer ImagePointerType;

  ImagePointerType image1 = as< ImageType >( in_image1 );
  ImagePointerType image2 = as< ImageType >( in_image2 );
  unsigned int mibins = 20;

  bool useprincaxis = true;
  typedef typename itk::ImageMaskSpatialObject<ImageDimension>::ImageType maskimagetype;
  typename maskimagetype::Pointer mask = ITK_NULLPTR;


  if( image1.IsNotNull() & image2.IsNotNull() )
    {
    typedef typename itk::ImageMomentsCalculator<ImageType> ImageCalculatorType;
    typedef itk::AffineTransform<RealType, ImageDimension> AffineType0;
    typedef typename ImageCalculatorType::MatrixType       MatrixType;
    typedef itk::Vector<float, ImageDimension>  VectorType;
    VectorType ccg1;
    VectorType cpm1;
    MatrixType cpa1;
    VectorType ccg2;
    VectorType cpm2;
    MatrixType cpa2;
    typename ImageCalculatorType::Pointer calculator1 =
      ImageCalculatorType::New();
    typename ImageCalculatorType::Pointer calculator2 =
      ImageCalculatorType::New();
    calculator1->SetImage(  image1 );
    calculator2->SetImage(  image2 );
    typename ImageCalculatorType::VectorType fixed_center;
    fixed_center.Fill(0);
    typename ImageCalculatorType::VectorType moving_center;
    moving_center.Fill(0);
    try
      {
      calculator1->Compute();
      fixed_center = calculator1->GetCenterOfGravity();
      ccg1 = calculator1->GetCenterOfGravity();
      cpm1 = calculator1->GetPrincipalMoments();
      cpa1 = calculator1->GetPrincipalAxes();
      try
        {
        calculator2->Compute();
        moving_center = calculator2->GetCenterOfGravity();
        ccg2 = calculator2->GetCenterOfGravity();
        cpm2 = calculator2->GetPrincipalMoments();
        cpa2 = calculator2->GetPrincipalAxes();
        }
      catch( ... )
        {
        fixed_center.Fill(0);
        }
      }
    catch( ... )
      {
      }
    if ( std::abs( bestscale - 1.0 ) < 1.e-6 )
      {
      RealType volelt1 = 1;
      RealType volelt2 = 1;
      for ( unsigned int d=0; d<ImageDimension; d++)
        {
        volelt1 *= image1->GetSpacing()[d];
        volelt2 *= image2->GetSpacing()[d];
        }
      bestscale =
        ( calculator2->GetTotalMass() * volelt2 )/
        ( calculator1->GetTotalMass() * volelt1 );
      RealType powlev = 1.0 / static_cast<RealType>(ImageDimension);
      bestscale = std::pow( bestscale , powlev );
    }
    unsigned int eigind1 = 1;
    unsigned int eigind2 = 1;
    if( ImageDimension == 3 )
      {
      eigind1 = 2;
      }
    typedef vnl_vector<RealType> EVectorType;
    typedef vnl_matrix<RealType> EMatrixType;
    EVectorType evec1_2ndary = cpa1.GetVnlMatrix().get_row( eigind2 );
    EVectorType evec1_primary = cpa1.GetVnlMatrix().get_row( eigind1 );
    EVectorType evec2_2ndary  = cpa2.GetVnlMatrix().get_row( eigind2 );
    EVectorType evec2_primary = cpa2.GetVnlMatrix().get_row( eigind1 );
    /** Solve Wahba's problem http://en.wikipedia.org/wiki/Wahba%27s_problem */
    EMatrixType B = outer_product( evec2_primary, evec1_primary );
    if( ImageDimension == 3 )
      {
      B = outer_product( evec2_2ndary, evec1_2ndary )
        + outer_product( evec2_primary, evec1_primary );
      }
    vnl_svd<RealType>    wahba( B );
    vnl_matrix<RealType> A_solution = wahba.V() * wahba.U().transpose();
    A_solution = vnl_inverse( A_solution );
    RealType det = vnl_determinant( A_solution  );
    if( ( det < 0 ) )
      {
      vnl_matrix<RealType> id( A_solution );
      id.set_identity();
      for( unsigned int i = 0; i < ImageDimension; i++ )
        {
        if( A_solution( i, i ) < 0 )
          {
          id( i, i ) = -1.0;
          }
        }
      A_solution =  A_solution * id.transpose();
      }
    if ( doReflection == 1 ||  doReflection == 3 )
      {
        vnl_matrix<RealType> id( A_solution );
        id.set_identity();
        id = id - 2.0 * outer_product( evec2_primary , evec2_primary  );
        A_solution = A_solution * id;
      }
    if ( doReflection > 1 )
      {
        vnl_matrix<RealType> id( A_solution );
        id.set_identity();
        id = id - 2.0 * outer_product( evec1_primary , evec1_primary  );
        A_solution = A_solution * id;
      }
    typename AffineType::Pointer affine1 = AffineType::New();
    typename AffineType::OffsetType trans = affine1->GetOffset();
    itk::Point<RealType, ImageDimension> trans2;
    for( unsigned int i = 0; i < ImageDimension; i++ )
      {
      trans[i] = moving_center[i] - fixed_center[i];
      trans2[i] =  fixed_center[i] * ( 1 );
      }
    affine1->SetIdentity();
    affine1->SetOffset( trans );
    if( useprincaxis )
      {
      affine1->SetMatrix( A_solution );
      }
    affine1->SetCenter( trans2 );
    if( ImageDimension > 3  )
      {
            std::vector<std::vector<float> > outMat(1, std::vector<float>(1));
      outMat[ 0 ][ 0 ] = 0;
      return outMat;
      }
    vnl_vector<RealType> evec_tert;
    if( ImageDimension == 3 )
      { // try to rotate around tertiary and secondary axis
      evec_tert = vnl_cross_3d( evec1_primary, evec1_2ndary );
      }
    if( ImageDimension == 2 )
      { // try to rotate around tertiary and secondary axis
      evec_tert = evec1_2ndary;
      evec1_2ndary = evec1_primary;
      }
    itk::Vector<RealType, ImageDimension> axis1;
    itk::Vector<RealType, ImageDimension> axis2;
    itk::Vector<RealType, ImageDimension> axis3;
    for( unsigned int d = 0; d < ImageDimension; d++ )
      {
      axis1[d] = evec_tert[d];
      axis2[d] = evec1_2ndary[d];
      axis3[d] = evec1_primary[d];
      }
    typename AffineType::Pointer simmer = AffineType::New();
    simmer->SetIdentity();
    simmer->SetCenter( trans2 );
    simmer->SetOffset( trans );
    typename AffineType0::Pointer affinesearch = AffineType0::New();
    affinesearch->SetIdentity();
    affinesearch->SetCenter( trans2 );
    typedef  itk::MultiStartOptimizerv4         OptimizerType;
    typename OptimizerType::MetricValuesListType metricvalues;
    typename OptimizerType::Pointer  mstartOptimizer = OptimizerType::New();
    typedef itk::CorrelationImageToImageMetricv4
      <ImageType, ImageType, ImageType> GCMetricType;
    typedef itk::MattesMutualInformationImageToImageMetricv4
      <ImageType, ImageType, ImageType> MetricType;
    typename MetricType::ParametersType newparams(  affine1->GetParameters() );
    typename GCMetricType::Pointer gcmetric = GCMetricType::New();
    gcmetric->SetFixedImage( image1 );
    gcmetric->SetVirtualDomainFromImage( image1 );
    gcmetric->SetMovingImage( image2 );
    gcmetric->SetMovingTransform( simmer );
    gcmetric->SetParameters( newparams );
    typename MetricType::Pointer mimetric = MetricType::New();
    mimetric->SetNumberOfHistogramBins( mibins );
    mimetric->SetFixedImage( image1 );
    mimetric->SetMovingImage( image2 );
    mimetric->SetMovingTransform( simmer );
    mimetric->SetParameters( newparams );
    if( mask.IsNotNull() )
      {
      typename itk::ImageMaskSpatialObject<ImageDimension>::Pointer so =
        itk::ImageMaskSpatialObject<ImageDimension>::New();
      so->SetImage( const_cast<maskimagetype *>( mask.GetPointer() ) );
      mimetric->SetFixedImageMask( so );
      gcmetric->SetFixedImageMask( so );
      }
    typedef  itk::ConjugateGradientLineSearchOptimizerv4 LocalOptimizerType;
    typename LocalOptimizerType::Pointer  localoptimizer =
      LocalOptimizerType::New();
    RealType     localoptimizerlearningrate = 0.1;
    localoptimizer->SetLearningRate( localoptimizerlearningrate );
    localoptimizer->SetMaximumStepSizeInPhysicalUnits(
      localoptimizerlearningrate );
    localoptimizer->SetNumberOfIterations( localSearchIterations );
    localoptimizer->SetLowerLimit( 0 );
    localoptimizer->SetUpperLimit( 2 );
    localoptimizer->SetEpsilon( 0.1 );
    localoptimizer->SetMaximumLineSearchIterations( 50 );
    localoptimizer->SetDoEstimateLearningRateOnce( true );
    localoptimizer->SetMinimumConvergenceValue( 1.e-6 );
    localoptimizer->SetConvergenceWindowSize( 5 );
    if( true )
      {
      typedef typename MetricType::FixedSampledPointSetType PointSetType;
      typedef typename PointSetType::PointType              PointType;
      typename PointSetType::Pointer      pset(PointSetType::New());
      unsigned int ind=0;
      unsigned int ct=0;
      itk::ImageRegionIteratorWithIndex<ImageType> It(image1,
        image1->GetLargestPossibleRegion() );
      for( It.GoToBegin(); !It.IsAtEnd(); ++It )
        {
        // take every N^th point
        if ( ct % 10 == 0  )
          {
          PointType pt;
          image1->TransformIndexToPhysicalPoint( It.GetIndex(), pt);
          pset->SetPoint(ind, pt);
          ind++;
          }
          ct++;
        }
      mimetric->SetFixedSampledPointSet( pset );
      mimetric->SetUseSampledPointSet( true );
      gcmetric->SetFixedSampledPointSet( pset );
      gcmetric->SetUseSampledPointSet( true );
    }
    if ( whichMetric.compare("MI") == 0  ) {
      mimetric->Initialize();
      typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType>
      RegistrationParameterScalesFromPhysicalShiftType;
      typename RegistrationParameterScalesFromPhysicalShiftType::Pointer
      shiftScaleEstimator =
      RegistrationParameterScalesFromPhysicalShiftType::New();
      shiftScaleEstimator->SetMetric( mimetric );
      shiftScaleEstimator->SetTransformForward( true );
      typename RegistrationParameterScalesFromPhysicalShiftType::ScalesType
      movingScales( simmer->GetNumberOfParameters() );
      shiftScaleEstimator->EstimateScales( movingScales );
      mstartOptimizer->SetScales( movingScales );
      mstartOptimizer->SetMetric( mimetric );
      localoptimizer->SetMetric( mimetric );
      localoptimizer->SetScales( movingScales );
    }
    if ( whichMetric.compare("MI") != 0  ) {
      gcmetric->Initialize();
      typedef itk::RegistrationParameterScalesFromPhysicalShift<GCMetricType>
        RegistrationParameterScalesFromPhysicalShiftType;
      typename RegistrationParameterScalesFromPhysicalShiftType::Pointer
        shiftScaleEstimator =
        RegistrationParameterScalesFromPhysicalShiftType::New();
      shiftScaleEstimator->SetMetric( gcmetric );
      shiftScaleEstimator->SetTransformForward( true );
      typename RegistrationParameterScalesFromPhysicalShiftType::ScalesType
      movingScales( simmer->GetNumberOfParameters() );
      shiftScaleEstimator->EstimateScales( movingScales );
      mstartOptimizer->SetScales( movingScales );
      mstartOptimizer->SetMetric( gcmetric );
      localoptimizer->SetMetric( gcmetric );
      localoptimizer->SetScales( movingScales );
    }
    typename OptimizerType::ParametersListType parametersList =
      mstartOptimizer->GetParametersList();
    affinesearch->SetIdentity();
    affinesearch->SetCenter( trans2 );
    affinesearch->SetOffset( trans );
    for ( unsigned int i = 0; i < thetas.size(); i++ )
      {
      RealType ang1 = thetas[i];
      RealType ang2 = 0; // FIXME should be psi
      RealType ang3 = 0; // FIXME should be psi
      if( ImageDimension == 3 )
        {
        for ( unsigned int jj = 0; jj < thetas2.size(); jj++ )
        {
        ang2=thetas2[jj];
        for ( unsigned int kk = 0; kk < thetas3.size(); kk++ )
        {
        ang3=thetas3[kk];
        affinesearch->SetIdentity();
        affinesearch->SetCenter( trans2 );
        affinesearch->SetOffset( trans );
        if( useprincaxis )
          {
          affinesearch->SetMatrix( A_solution );
          }
        affinesearch->Rotate3D(axis1, ang1, 1);
        affinesearch->Rotate3D(axis2, ang2, 1);
        affinesearch->Rotate3D(axis3, ang3, 1);
//        affinesearch->Scale( bestscale ); // BUG: annoying, cant do for rigid
        simmer->SetMatrix(  affinesearch->GetMatrix() );
        parametersList.push_back( simmer->GetParameters() );
        } // kk
        }
        }
      if( ImageDimension == 2 )
        {
        affinesearch->SetIdentity();
        affinesearch->SetCenter( trans2 );
        affinesearch->SetOffset( trans );
        if( useprincaxis )
          {
          affinesearch->SetMatrix( A_solution );
          }
        affinesearch->Rotate2D( ang1, 1);
//        affinesearch->Scale( bestscale );
        simmer->SetMatrix(  affinesearch->GetMatrix() );
        typename AffineType::ParametersType pp =
          simmer->GetParameters();
        parametersList.push_back( simmer->GetParameters() );
        }
      }
    mstartOptimizer->SetParametersList( parametersList );
    if( localSearchIterations > 0 )
      {
      mstartOptimizer->SetLocalOptimizer( localoptimizer );
      }
    mstartOptimizer->StartOptimization();
    typename AffineType::Pointer bestaffine = AffineType::New();
    bestaffine->SetCenter( trans2 );
    bestaffine->SetParameters( mstartOptimizer->GetBestParameters() );
    if ( txfn.length() > 3 )
      {
      typename AffineType::Pointer bestaffine = AffineType::New();
      bestaffine->SetCenter( trans2 );
      bestaffine->SetParameters( mstartOptimizer->GetBestParameters() );
      typedef itk::TransformFileWriter TransformWriterType;
      typename TransformWriterType::Pointer transformWriter =
        TransformWriterType::New();
      transformWriter->SetInput( bestaffine );
      transformWriter->SetFileName( txfn.c_str() );
      transformWriter->Update();
      }
    metricvalues = mstartOptimizer->GetMetricValuesList();
    unsigned int ncols = mstartOptimizer->GetBestParameters().Size() +
      1 + ImageDimension;

    std::vector<std::vector<float> > outMat(metricvalues.size(), std::vector<float>(ncols));
    for ( unsigned int k = 0; k < metricvalues.size(); k++ )
      {
      outMat[ k ][ 0 ] = metricvalues[ k ];
      typename MetricType::ParametersType resultParams =
        mstartOptimizer->GetParametersList( )[ k ];
      for ( unsigned int kp = 0; kp < resultParams.Size(); kp++ )
        {
        outMat[ k ][ kp + 1 ] = resultParams[ kp ];
        }
      // set the fixed parameters
      unsigned int baseind = resultParams.Size() + 1;
      for ( unsigned int kp = 0; kp < ImageDimension; kp++ )
        outMat[ k ][ baseind + kp ] = bestaffine->GetFixedParameters()[ kp ];
      }
    return outMat;
    }
  else
    {
      std::vector<std::vector<float> > outMat(1, std::vector<float>(1));
      outMat[ 0 ][ 0 ] = 0;
      return outMat;
    }
}


template <class PixelType, unsigned int Dimension>
py::capsule convolveImage( py::capsule ants_image,
                                    py::capsule ants_kernel )
{
  typedef itk::Image<PixelType,Dimension> ImageType;
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType image = as< ImageType >( ants_image );
  ImagePointerType kernel = as< ImageType >( ants_kernel );

  typedef itk::ConvolutionImageFilter<ImageType> FilterType;
  typename FilterType::Pointer convolutionFilter = FilterType::New();
  convolutionFilter->SetInput( image );
  convolutionFilter->SetKernelImage( kernel );
  convolutionFilter->Update();

  py::capsule convOutput = wrap< ImageType >( convolutionFilter->GetOutput() );
  return convOutput;

}

PYBIND11_MODULE(invariantImageSimilarity, m)
{
  // whichTx : 0=affine, 1 = similarity, 2 = rigid
  m.def("invariantImageSimilarity_Affine2D", &invariantImageSimilarity<itk::AffineTransform<double,2>,2>);
  m.def("invariantImageSimilarity_Affine3D", &invariantImageSimilarity<itk::AffineTransform<double,3>,3>);
  m.def("invariantImageSimilarity_Similarity2D", &invariantImageSimilarity<itk::Similarity2DTransform<double>,2>);
  m.def("invariantImageSimilarity_Similarity3D", &invariantImageSimilarity<itk::Similarity3DTransform<double>,3>);
  m.def("invariantImageSimilarity_Rigid2D", &invariantImageSimilarity<itk::Euler2DTransform<double>,2>);
  m.def("invariantImageSimilarity_Rigid3D", &invariantImageSimilarity<itk::Euler3DTransform<double>,3>);

  m.def("convolveImageF2", &convolveImage<float, 2>);
  m.def("convolveImageF3", &convolveImage<float, 3>);
  m.def("convolveImageF4", &convolveImage<float, 4>);

}
