
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
#include "itkCastImageFilter.h"
#include "antscore/itkSimulatedBSplineDisplacementFieldSource.h"
#include "antscore/itkSimulatedExponentialDisplacementFieldSource.h"
#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template<class PrecisionType, unsigned int Dimension>
AntsImage<itk::VectorImage<PrecisionType, Dimension>> simulateBsplineDisplacementField(AntsImage<itk::Image<PrecisionType, Dimension>> & antsDomainImage,
                                             unsigned int numberOfRandomPoints,
                                             float standardDeviationDisplacementField,
                                             bool enforceStationaryBoundary,
                                             unsigned int numberOfFittingLevels,
                                             std::vector<unsigned int> numberOfControlPoints)
{
  using ImageType = itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType domainImage = antsDomainImage.ptr;

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
  antsField->AllocateInitialized();

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

  AntsImage<ANTsFieldType> outImage = { antsField };
  return outImage;
}

template<class PrecisionType, unsigned int Dimension>
AntsImage<itk::VectorImage<PrecisionType, Dimension>> simulateExponentialDisplacementField(AntsImage<itk::Image<PrecisionType, Dimension>> & antsDomainImage,
                                                 unsigned int numberOfRandomPoints,
                                                 float standardDeviationDisplacementField,
                                                 bool enforceStationaryBoundary,
                                                 float standardDeviationSmoothing)
{
  using ImageType = itk::Image<PrecisionType, Dimension>;
  using ImagePointerType = typename ImageType::Pointer;

  ImagePointerType domainImage = antsDomainImage.ptr;

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
  antsField->AllocateInitialized();

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

  AntsImage<ANTsFieldType> outImage = { antsField };
  return outImage;
}

void local_simulateDisplacementField(nb::module_ &m)
{
  m.def("simulateBsplineDisplacementField2D", &simulateBsplineDisplacementField<float, 2>);
  m.def("simulateBsplineDisplacementField3D", &simulateBsplineDisplacementField<float, 3>);

  m.def("simulateExponentialDisplacementField2D", &simulateExponentialDisplacementField<float, 2>);
  m.def("simulateExponentialDisplacementField3D", &simulateExponentialDisplacementField<float, 3>);
}

