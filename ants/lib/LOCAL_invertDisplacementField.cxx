
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkVector.h"
#include "itkInvertDisplacementFieldImageFilter.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <unsigned int Dimension>
py::capsule invertDisplacementField( py::capsule & antsDisplacementField,
                                     py::capsule & antsInverseFieldInitialEstimate,
                                     unsigned int maximumNumberOfIterations,
                                     float meanErrorToleranceThreshold,
                                     float maxErrorToleranceThreshold,
                                     bool enforceBoundaryCondition )
{
  using RealType = float;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;

  using ITKFieldType = itk::Image<VectorType, Dimension>;
  using ITKFieldPointerType = typename ITKFieldType::Pointer;

  using IteratorType = itk::ImageRegionConstIteratorWithIndex<ITKFieldType>;

  ITKFieldPointerType itkDisplacementField = as< ITKFieldType >( antsDisplacementField );
  ITKFieldPointerType itkInverseFieldInitialEstimate = as< ITKFieldType >( antsInverseFieldInitialEstimate );

  typedef itk::InvertDisplacementFieldImageFilter<ITKFieldType> InverterType;
  typename InverterType::Pointer inverter = InverterType::New();

  inverter->SetInput( itkDisplacementField );
  inverter->SetInverseFieldInitialEstimate( itkInverseFieldInitialEstimate );
  inverter->SetMaximumNumberOfIterations( maximumNumberOfIterations );
  inverter->SetMeanErrorToleranceThreshold( meanErrorToleranceThreshold );
  inverter->SetMaxErrorToleranceThreshold( maxErrorToleranceThreshold );
  inverter->SetEnforceBoundaryCondition( enforceBoundaryCondition );

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( inverter->GetOutput() );
  antsField->SetRegions( inverter->GetOutput()->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->Allocate();

  IteratorType ItI( inverter->GetOutput(),
    inverter->GetOutput()->GetRequestedRegion() );
  for( ItI.GoToBegin(); !ItI.IsAtEnd(); ++ItI )
    {
    VectorType data = ItI.Value();

    typename ANTsFieldType::PixelType antsVector( Dimension );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      antsVector[d] = data[d];
      }
    antsField->SetPixel( ItI.GetIndex(), antsVector );
    }

  return wrap< ANTsFieldType >( antsField );
}

PYBIND11_MODULE(invertDisplacementField, m)
{
  m.def("invertDisplacementFieldF2", &invertDisplacementField<2>);
  m.def("invertDisplacementFieldF3", &invertDisplacementField<3>);
}

