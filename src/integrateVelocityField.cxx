
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
#include "itkImageFileWriter.h"
#include "itkVector.h"
#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

template <unsigned int Dimension>
AntsImage<itk::VectorImage<float, Dimension>> integrateVelocityField( AntsImage<itk::VectorImage<float, Dimension+1>> & antsVelocityField,
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

  ANTsVelocityFieldPointerType inputVelocityField = antsVelocityField.ptr;

  ITKVelocityFieldPointerType inputITKVelocityField = ITKVelocityFieldType::New();
  inputITKVelocityField->CopyInformation( inputVelocityField );
  inputITKVelocityField->SetRegions( inputVelocityField->GetRequestedRegion() );
  inputITKVelocityField->AllocateInitialized();

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
  antsField->AllocateInitialized();

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

  AntsImage<ANTsFieldType> out_ants_image = { antsField };
  return out_ants_image;
}

void local_integrateVelocityField(nb::module_ &m)
{
  m.def("integrateVelocityFieldD2", &integrateVelocityField<2>);
  m.def("integrateVelocityFieldD3", &integrateVelocityField<3>);
}

