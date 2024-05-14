
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

#include "itkAffineTransform.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageMomentsCalculator.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileWriter.h"
#include "vnl/vnl_inverse.h"

#include "antsImage.h"

namespace nb = nanobind;
using namespace nb::literals;

/*
template< unsigned int ImageDimension >
int antsReoHelper(
  typename itk::Image< float , ImageDimension >::Pointer image1,
  std::string txfn, std::vector<int> axis , std::vector<int> axis2,
  std::vector<int> doReflection, std::vector<int> doScale )
{
  typedef double RealType;
  typedef itk::Image< float , ImageDimension > ImageType;

  typedef typename itk::ImageMomentsCalculator<ImageType> ImageCalculatorType;
  typedef itk::AffineTransform<RealType, ImageDimension> AffineType;
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
  calculator1->SetImage(  image1 );
  typename ImageCalculatorType::VectorType fixed_center;
  fixed_center.Fill(0);

  // was a try-catch block around this in ANTsR
  calculator1->Compute();
  fixed_center = calculator1->GetCenterOfGravity();
  ccg1 = calculator1->GetCenterOfGravity();
  cpm1 = calculator1->GetPrincipalMoments();
  cpa1 = calculator1->GetPrincipalAxes();

  unsigned int eigind1 = 1;
  unsigned int eigind2 = 1;
  typedef vnl_vector<RealType> EVectorType;
  typedef vnl_matrix<RealType> EMatrixType;
  EVectorType evec1_2ndary = cpa1.GetVnlMatrix().get_row( eigind2 );
  EVectorType evec1_primary = cpa1.GetVnlMatrix().get_row( eigind1 );
  EVectorType evec2_2ndary;
  evec2_2ndary.set_size( ImageDimension );
  evec2_2ndary.fill(0);
  EVectorType evec2_primary;
  evec2_primary.set_size( ImageDimension );
  evec2_primary.fill(0);
  for ( unsigned int i = 0; i < ImageDimension; i++ )
    evec2_primary[i] = axis[i];
  // Solve Wahba's problem http://en.wikipedia.org/wiki/Wahba%27s_problem
  EMatrixType B = outer_product( evec2_primary, evec1_primary );
  if( ImageDimension == 3 )
    {
    for ( unsigned int i = 0; i < ImageDimension; i++ )
        evec2_primary[i] = axis2[i];
    B = outer_product( evec2_2ndary, evec1_2ndary )
      + outer_product( evec2_primary, evec1_primary );
    }
    vnl_svd<RealType>    wahba( B );
    vnl_matrix<RealType> A_solution = wahba.V() * wahba.U().transpose();
    A_solution = vnl_inverse( A_solution );
    RealType det = vnl_determinant( A_solution  );
    if( det < 0 )
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
    if ( doReflection[0] == 1 ||  doReflection[0] == 3 )
      {
      vnl_matrix<RealType> id( A_solution );
      id.set_identity();
      id = id - 2.0 * outer_product( evec2_primary , evec2_primary  );
      A_solution = A_solution * id;
      }
    if ( doReflection[0] > 1 )
      {
      vnl_matrix<RealType> id( A_solution );
      id.set_identity();
      id = id - 2.0 * outer_product( evec1_primary , evec1_primary  );
      A_solution = A_solution * id;
      }
    if ( doScale[0] > 0 )
      {
      vnl_matrix<RealType> id( A_solution );
      id.set_identity();
      id = id * doScale[0];
      A_solution = A_solution * id;
      }
    det = vnl_determinant( A_solution  );
    std::cout << " det " << det << std::endl;
    std::cout << " A_solution " << std::endl;
    std::cout << A_solution << std::endl;
    typename AffineType::Pointer affine1 = AffineType::New();
    typename AffineType::OffsetType trans = affine1->GetOffset();
    itk::Point<RealType, ImageDimension> trans2;
    trans2.Fill(0);
    for( unsigned int i = 0; i < ImageDimension; i++ )
      {
      trans2[i] =  fixed_center[i] * ( 1 );
      }
    affine1->SetIdentity();
    affine1->SetOffset( trans );
    affine1->SetMatrix( A_solution );
    affine1->SetCenter( trans2 );
    // write tx
    typedef itk::TransformFileWriter TransformWriterType;
    typename TransformWriterType::Pointer transformWriter = TransformWriterType::New();
    transformWriter->SetInput( affine1 );
    transformWriter->SetFileName( txfn.c_str() );
    transformWriter->Update();

  return 0;

}

template <typename ImageType, unsigned int Dimension>
int reorientImage(  py::capsule in_image, std::string txfn,
                      std::vector<int> axis1, std::vector<int> axis2,
                      std::vector<int> rrfl, std::vector<int> rscl )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itk_image = as<ImageType>( in_image );
  antsReoHelper<Dimension>( itk_image, txfn, axis1, axis2, rrfl, rscl );

  return 0;
}

*/


template <typename ImageType, unsigned int Dimension>
std::vector<double> centerOfMass( AntsImage<ImageType> & image  )
{
  typedef typename ImageType::Pointer ImagePointerType;
  ImagePointerType itkimage = image.ptr;

  typedef typename itk::ImageMomentsCalculator<ImageType> ImageCalculatorType;
  typename ImageCalculatorType::VectorType com( Dimension );
  com.Fill( 0 );

  std::vector<double> myCoM( Dimension );

  typename ImageCalculatorType::Pointer calculator1 = ImageCalculatorType::New();
  calculator1->SetImage( itkimage );
  calculator1->Compute();
  com = calculator1->GetCenterOfGravity();

  for ( unsigned int k = 0; k < Dimension; k++ )
  {
    myCoM[ k ] = com[ k ];
  }

  return myCoM;
}

void local_reorientImage(nb::module_ &m)
{
//  m.def("reorientImageF2", &reorientImage<itk::Image<float, 2>, 2>);
//  m.def("reorientImageF3", &reorientImage<itk::Image<float, 3>, 3>);
//  m.def("reorientImageF4", &reorientImage<itk::Image<float, 4>, 4>);

  m.def("centerOfMass", &centerOfMass<itk::Image<float, 2>, 2>);
  m.def("centerOfMass", &centerOfMass<itk::Image<float, 3>, 3>);
  m.def("centerOfMass", &centerOfMass<itk::Image<float, 4>, 4>);
}
