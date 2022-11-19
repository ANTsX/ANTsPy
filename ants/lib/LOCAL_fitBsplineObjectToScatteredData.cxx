
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkPointSet.h"
#include "itkBSplineScatteredDataPointSetToImageFilter.h"
#include "itkVectorIndexSelectionCastImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template<unsigned int DataDimension>
py::array_t<double> fitBsplineCurveHelper(
  py::array_t<double> scatteredData,
  py::array_t<double> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  const unsigned int ParametricDimension = 1;

  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData.unchecked<2>();
  auto parametricDataR = parametricData.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.shape(0);

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR(n, d);
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR(n, d);
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  py::array_t<double> bsplineCurve({ parametricDomainSize[0], DataDimension });
  auto bsplineCurveR = bsplineCurve.mutable_unchecked<2>();

  unsigned int count = 0;
  IteratorType It( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    ScatteredDataType data = It.Value();
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      bsplineCurveR(count, d) = data[d];
      }
    count++;
    }

  return bsplineCurve;
}

template<unsigned int ParametricDimension>
py::capsule fitBsplineImageHelper(
  py::array_t<double> scatteredData,
  py::array_t<double> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  const unsigned int DataDimension = 1;

  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData.unchecked<2>();
  auto parametricDataR = parametricData.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.shape(0);

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR(n, d);
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR(n, d);
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  using ScalarImageType = itk::Image<RealType, ParametricDimension>;
  using SelectionFilterType = itk::VectorIndexSelectionCastImageFilter<OutputImageType, ScalarImageType>;
  typename SelectionFilterType::Pointer selectionFilter = SelectionFilterType::New();
  selectionFilter->SetIndex( 0 );
  selectionFilter->SetInput( bsplineFilter->GetOutput() );
  selectionFilter->Update();

  return wrap< ScalarImageType >( selectionFilter->GetOutput() );
}

template<unsigned int ParametricDimension, unsigned int DataDimension>
py::capsule fitBsplineVectorImageHelper(
  py::array_t<double> scatteredData,
  py::array_t<double> parametricData,
  std::vector<double> dataWeights,
  std::vector<double> parametricDomainOrigin,
  std::vector<double> parametricDomainSpacing,
  std::vector<unsigned int> parametricDomainSize,
  std::vector<bool> isParametricDimensionClosed,
  unsigned int numberOfFittingLevels,
  std::vector<unsigned int> numberOfControlPoints,
  unsigned int splineOrder )
{
  using RealType = float;
  using ScatteredDataType = itk::Vector<RealType, DataDimension>;
  using PointSetType = itk::PointSet<ScatteredDataType, ParametricDimension>;
  using OutputImageType = itk::Image<ScatteredDataType, ParametricDimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<OutputImageType>;

  using BSplineFilterType = itk::BSplineScatteredDataPointSetToImageFilter<PointSetType, OutputImageType>;
  using WeightsContainerType = typename BSplineFilterType::WeightsContainerType;

  typename PointSetType::Pointer pointSet = PointSetType::New();
  pointSet->Initialize();
  typename WeightsContainerType::Pointer weights = WeightsContainerType::New();

  auto scatteredDataR = scatteredData.unchecked<2>();
  auto parametricDataR = parametricData.unchecked<2>();

  unsigned int numberOfPoints = scatteredDataR.shape(0);

  for( unsigned int n = 0; n < numberOfPoints; n++ )
    {
    typename PointSetType::PointType point;
    for( unsigned int d = 0; d < ParametricDimension; d++ )
      {
      point[d] = parametricDataR(n, d);
      }
    pointSet->SetPoint( n, point );

    ScatteredDataType data( 0.0 );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      data[d] = scatteredDataR(n, d);
      }
    pointSet->SetPointData( n, data );

    weights->InsertElement( n, dataWeights[n] );
    }

  typename BSplineFilterType::Pointer bsplineFilter = BSplineFilterType::New();
  bsplineFilter->SetInput( pointSet );
  bsplineFilter->SetPointWeights( weights );
  bsplineFilter->SetGenerateOutputImage( true );

  typename OutputImageType::PointType origin;
  typename OutputImageType::SpacingType spacing;
  typename OutputImageType::SizeType size;
  typename BSplineFilterType::ArrayType ncps;
  typename BSplineFilterType::ArrayType isClosed;

  for( unsigned int d = 0; d < ParametricDimension; d++ )
    {
    origin[d] = parametricDomainOrigin[d];
    spacing[d] = parametricDomainSpacing[d];
    size[d] = parametricDomainSize[d];
    ncps[d] = numberOfControlPoints[d];
    isClosed[d] = static_cast<bool>( isParametricDimensionClosed[d] );
    }
  bsplineFilter->SetOrigin( origin );
  bsplineFilter->SetSpacing( spacing );
  bsplineFilter->SetSize( size );
  bsplineFilter->SetNumberOfControlPoints( ncps );
  bsplineFilter->SetSplineOrder( splineOrder );
  bsplineFilter->SetNumberOfLevels( numberOfFittingLevels );
  bsplineFilter->SetCloseDimension( isClosed );
  bsplineFilter->Update();

  //////////////////////////
  //
  //  Only difference between the Curve, Image, and Object function
  //  is the return type.
  //

  using VectorImageType = itk::VectorImage<RealType, ParametricDimension>;
  using VectorImagePointerType = typename VectorImageType::Pointer;

  typename VectorImageType::DirectionType direction;
  direction.SetIdentity();

  VectorImagePointerType antsField = VectorImageType::New();
  antsField->SetOrigin( origin );
  antsField->SetRegions( size );
  antsField->SetSpacing( spacing );
  antsField->SetVectorLength( DataDimension );
  antsField->SetDirection( direction );
  antsField->Allocate();

  IteratorType It( bsplineFilter->GetOutput(),
    bsplineFilter->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    ScatteredDataType data = It.Value();

    typename VectorImageType::PixelType antsVector( DataDimension );
    for( unsigned int d = 0; d < DataDimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  return wrap< VectorImageType >( antsField );
}

PYBIND11_MODULE(fitBsplineObjectToScatteredData, m)
{
  m.def("fitBsplineObjectToScatteredDataP1D1", &fitBsplineCurveHelper<1>);
  m.def("fitBsplineObjectToScatteredDataP1D2", &fitBsplineCurveHelper<2>);
  m.def("fitBsplineObjectToScatteredDataP1D3", &fitBsplineCurveHelper<3>);
  m.def("fitBsplineObjectToScatteredDataP1D4", &fitBsplineCurveHelper<4>);

  m.def("fitBsplineObjectToScatteredDataP2D1", &fitBsplineImageHelper<2>);
  m.def("fitBsplineObjectToScatteredDataP3D1", &fitBsplineImageHelper<3>);
  m.def("fitBsplineObjectToScatteredDataP4D1", &fitBsplineImageHelper<4>);

  m.def("fitBsplineObjectToScatteredDataP2D2", &fitBsplineVectorImageHelper<2, 2>);
  m.def("fitBsplineObjectToScatteredDataP3D3", &fitBsplineVectorImageHelper<3, 3>);

  m.def("fitBsplineObjectToScatteredDataP3D2", &fitBsplineVectorImageHelper<3, 2>);
  m.def("fitBsplineObjectToScatteredDataP4D3", &fitBsplineVectorImageHelper<4, 3>);
}
