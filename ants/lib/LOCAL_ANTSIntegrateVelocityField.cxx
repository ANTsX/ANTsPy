#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <algorithm>
#include <vector>
#include "antscore/antsUtilities.h"
#include "antscore/antsAllocImage.h"
#include <algorithm>

#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkTimeVaryingVelocityFieldIntegrationImageFilter.h"
#include "itkTimeVaryingVelocityFieldTransform.h"
#include "itkImageFileWriter.h"
#include "ReadWriteData.h"
#include "itkImage.h"
#include "itkRescaleIntensityImageFilter.h"

// NEED THIS INCLUDE FOR WRAPPING/UNWRAPPING
#include "LOCAL_antsImage.h"

// NEED THIS FOR PYBIND11
namespace py = pybind11;


template< class ImageType >
void integrateVelocityField(
    py::capsule & r_refimg,
    std::string r_velocity,
    std::string r_deformation,
    float r_t0,
    float r_t1,
    float r_dt )
{
  using PixelType = typename ImageType::PixelType;
  enum { Dimension = ImageType::ImageDimension };
  PixelType starttime = static_cast< PixelType >( r_t0  );
  PixelType finishtime = static_cast< PixelType >( r_t1  );
  PixelType dT = static_cast< PixelType >( r_dt );
  using VectorType = itk::Vector<PixelType, Dimension>;

  typedef typename ImageType::Pointer       ImagePointerType;
  typename ImageType::Pointer input = as< ImageType >( r_refimg );

  using DisplacementFieldType = itk::Image<VectorType, Dimension>;
  using TimeVaryingVelocityFieldType = itk::Image<VectorType, Dimension + 1>;
  using tvt = TimeVaryingVelocityFieldType;
  typename tvt::Pointer timeVaryingVelocity;

  ReadImage<tvt>(timeVaryingVelocity, r_velocity.c_str() );

  VectorType zero;
  zero.Fill(0);
  typename DisplacementFieldType::Pointer deformation =
    AllocImage<DisplacementFieldType>(input, zero);

  if( starttime < 0 )
    {
    starttime = 0;
    }
  if( starttime > 1 )
    {
    starttime = 1;
    }
  if( finishtime < 0 )
    {
    finishtime = 0;
    }
  if( finishtime > 1 )
    {
    finishtime = 1;
    }

  using IntegratorType = itk::TimeVaryingVelocityFieldIntegrationImageFilter<TimeVaryingVelocityFieldType, DisplacementFieldType>;
  typename IntegratorType::Pointer integrator = IntegratorType::New();
  integrator->SetInput( timeVaryingVelocity );
  integrator->SetLowerTimeBound( starttime );
  integrator->SetUpperTimeBound( finishtime );
  integrator->SetNumberOfIntegrationSteps( static_cast<unsigned int>( std::round( 1.0 / dT ) ) );
  integrator->Update();
  WriteImage<DisplacementFieldType>( integrator->GetOutput(), r_deformation.c_str() );
  return;
}


// DECLARE OUR FUNCTION FOR APPROPRIATE TYPES
PYBIND11_MODULE(integrateVelocityField, m)
{
    m.def("integrateVelocityField2D", &integrateVelocityField<itk::Image<float,2>>);
    m.def("integrateVelocityField3D", &integrateVelocityField<itk::Image<float,3>>);
}
