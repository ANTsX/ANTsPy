
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "antscore/itkSimulatedBSplineDisplacementFieldSource.h"
#include "antscore/itkSimulatedExponentialDisplacementFieldSource.h"
#include "LOCAL_antsImage.h"

namespace py = pybind11;
using namespace py::literals;

template<class PrecisionType, unsigned int Dimension>
py::capsule simulateBsplineDisplacementField(py::capsule & antsDomainImage,
                                             unsigned int numberOfRandomPoints,
                                             float standardDeviationDisplacementField,
                                             bool enforceStationaryBoundary,
                                             unsigned int numberOfFittingLevels,
                                             std::vector<unsigned int> numberOfControlPoints)
{
  using ImageType = itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType domainImage = as< ImageType >( antsDomainImage );

  using VectorType = itk::Vector<PrecisionType, Dimension>;
  using DisplacementFieldType = itk::Image<VectorType, Dimension>;
  using ANTsFieldType = itk::VectorImage<PrecisionType, Dimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<DisplacementFieldType>;

  using BSplineSimulatorType = itk::SimulatedBSplineDisplacementFieldSource<DisplacementFieldType>;

  typename BSplineSimulatorType::ArrayType ncps;
  for( unsigned int d = 0; d < numberOfControlPoints.size(); ++d )
    {
    ncps = numberOfControlPoints[d];
    }

  using RealImageType = typename BSplineSimulatorType::RealImageType;
  using CastImageFilterType = itk::CastImageFilter<ImageType, RealImageType>;
  typename CastImageFilterType::Pointer caster = CastImageFilterType::New();
  caster->SetInput( domainImage );
  caster->Update();

  typename BSplineSimulatorType::Pointer bsplineSimulator = BSplineSimulatorType::New();
  bsplineSimulator->SetDisplacementFieldDomainFromImage( caster->GetOutput() );
  bsplineSimulator->SetNumberOfRandomPoints( numberOfRandomPoints );
  bsplineSimulator->SetEnforceStationaryBoundary( enforceStationaryBoundary );
  bsplineSimulator->SetDisplacementNoiseStandardDeviation( standardDeviationDisplacementField );
  bsplineSimulator->SetNumberOfFittingLevels( numberOfFittingLevels );
  bsplineSimulator->SetNumberOfControlPoints( ncps );
  bsplineSimulator->Update();

  typename ANTsFieldType::Pointer antsField = ANTsFieldType::New();
  antsField->CopyInformation( domainImage );
  antsField->SetRegions( domainImage->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->Allocate();

  IteratorType It( bsplineSimulator->GetOutput(), 
    bsplineSimulator->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    VectorType itkVector = It.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = itkVector[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  return wrap< ANTsFieldType >( antsField );
}

template<class PrecisionType, unsigned int Dimension>
py::capsule simulateExponentialDisplacementField(py::capsule & antsDomainImage,
                                                 unsigned int numberOfRandomPoints,
                                                 float standardDeviationDisplacementField,
                                                 bool enforceStationaryBoundary,
                                                 float standardDeviationSmoothing)
{
  using ImageType = itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType domainImage = as< ImageType >( antsDomainImage );

  using VectorType = itk::Vector<PrecisionType, Dimension>;
  using DisplacementFieldType = itk::Image<VectorType, Dimension>;
  using ANTsFieldType = itk::VectorImage<PrecisionType, Dimension>;
  using IteratorType = itk::ImageRegionIteratorWithIndex<DisplacementFieldType>;

  using ExponentialSimulatorType = itk::SimulatedExponentialDisplacementFieldSource<DisplacementFieldType>;

  using RealImageType = typename ExponentialSimulatorType::RealImageType;
  using CastImageFilterType = itk::CastImageFilter<ImageType, RealImageType>;
  typename CastImageFilterType::Pointer caster = CastImageFilterType::New();
  caster->SetInput( domainImage );
  caster->Update();

  typename ExponentialSimulatorType::Pointer exponentialSimulator = ExponentialSimulatorType::New();
  exponentialSimulator->SetDisplacementFieldDomainFromImage( caster->GetOutput() );
  exponentialSimulator->SetNumberOfRandomPoints( numberOfRandomPoints );
  exponentialSimulator->SetEnforceStationaryBoundary( enforceStationaryBoundary );
  exponentialSimulator->SetDisplacementNoiseStandardDeviation( standardDeviationDisplacementField );
  exponentialSimulator->SetSmoothingStandardDeviation( standardDeviationSmoothing );
  exponentialSimulator->Update();

  typename ANTsFieldType::Pointer antsField = ANTsFieldType::New();
  antsField->CopyInformation( domainImage );
  antsField->SetRegions( domainImage->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->Allocate();

  IteratorType It( exponentialSimulator->GetOutput(), 
    exponentialSimulator->GetOutput()->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    VectorType itkVector = It.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = itkVector[d];
      }
    antsField->SetPixel( It.GetIndex(), antsVector );
    }

  return wrap< ANTsFieldType >( antsField );
}

PYBIND11_MODULE(simulateDisplacementField, m)
{
  m.def("simulateBsplineDisplacementField2D", &simulateBsplineDisplacementField<float, 2>);
  m.def("simulateBsplineDisplacementField3D", &simulateBsplineDisplacementField<float, 3>);

  m.def("simulateExponentialDisplacementField2D", &simulateExponentialDisplacementField<float, 2>);
  m.def("simulateExponentialDisplacementField3D", &simulateExponentialDisplacementField<float, 3>);
}

