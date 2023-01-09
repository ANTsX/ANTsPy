
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkVector.h"
#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "LOCAL_antsImage.h"

namespace py = pybind11;

template <unsigned int Dimension>
py::capsule integrateVelocityField( py::capsule & antsVelocityField,
                                    float lowerBound,
                                    float upperBound,
                                    unsigned int numberOfIntegrationSteps )
{
  using RealType = float;

  using ANTsVelocityFieldType = itk::VectorImage<RealType, Dimension+1>;
  using ANTsVelocityFieldPointerType = typename ANTsVelocityFieldType::Pointer;

  using ANTsFieldType = itk::VectorImage<RealType, Dimension>;
  using ANTsFieldPointerType = typename ANTsFieldType::Pointer;

  using VectorType = itk::Vector<RealType, Dimension>;

  using ITKVelocityFieldType = itk::Image<VectorType, Dimension+1>;
  using ITKVelocityFieldPointerType = typename ITKVelocityFieldType::Pointer;
  using ITKFieldType = itk::Image<VectorType, Dimension>;

  using IteratorType = itk::ImageRegionIteratorWithIndex<ITKVelocityFieldType>;
  using ConstIteratorType = itk::ImageRegionConstIteratorWithIndex<ITKFieldType>;

  ANTsVelocityFieldPointerType inputVelocityField = as<ANTsVelocityFieldType>( antsVelocityField );

  ITKVelocityFieldPointerType inputITKVelocityField = ITKVelocityFieldType::New();
  inputITKVelocityField->CopyInformation( inputVelocityField );
  inputITKVelocityField->SetRegions( inputVelocityField->GetRequestedRegion() );
  inputITKVelocityField->Allocate();

  IteratorType It( inputITKVelocityField,
                   inputITKVelocityField->GetRequestedRegion() );
  for( It.GoToBegin(); !It.IsAtEnd(); ++It )
    {
    VectorType vector;

    typename ANTsFieldType::PixelType antsVector = inputVelocityField->GetPixel( It.GetIndex() );
    for( unsigned int d = 0; d < Dimension; d++ )
      {
      vector[d] = antsVector[d];
      }
    It.Set( vector );
    }

  using IntegratorType = itk::TimeVaryingVelocityFieldIntegrationImageFilter<ITKVelocityFieldType, ITKFieldType>;
  typename IntegratorType::Pointer integrator = IntegratorType::New();

  integrator->SetInput( inputITKVelocityField );
  integrator->SetLowerTimeBound( lowerBound );
  integrator->SetUpperTimeBound( upperBound );
  integrator->SetNumberOfIntegrationSteps( numberOfIntegrationSteps );
  integrator->Update();

  //////////////////////////
  //
  //  Now convert back to vector image type.
  //

  ANTsFieldPointerType antsField = ANTsFieldType::New();
  antsField->CopyInformation( integrator->GetOutput() );
  antsField->SetRegions( integrator->GetOutput()->GetRequestedRegion() );
  antsField->SetVectorLength( Dimension );
  antsField->Allocate();

  ConstIteratorType ItI( integrator->GetOutput(),
    integrator->GetOutput()->GetRequestedRegion() );
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

PYBIND11_MODULE(integrateVelocityField, m)
{
  m.def("integrateVelocityFieldD2", &integrateVelocityField<2>);
  m.def("integrateVelocityFieldD3", &integrateVelocityField<3>);
}

