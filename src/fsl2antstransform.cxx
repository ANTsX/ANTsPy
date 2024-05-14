
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/list.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <vector>
#include <string>

#include "itkImage.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkCastImageFilter.h"
#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_diag_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_det.h"
#include "vnl/vnl_inverse.h"
#include "vnl/algo/vnl_real_eigensystem.h"
#include "vnl/algo/vnl_qr.h"

#include "antsTransform.h"
#include "antsImage.h"

#define RAS_TO_FSL 0
#define FSL_TO_RAS 1

namespace nb = nanobind;
using namespace nb::literals;

/**
 * Get a matrix that maps points voxel coordinates to RAS coordinates
 */
template< class ImageType, class TransformMatrixType >
TransformMatrixType GetVoxelSpaceToRASPhysicalSpaceMatrix(typename ImageType::Pointer image)
  {
  // Generate intermediate terms
  vnl_matrix_fixed<double, 3U, 3U> m_dir, m_ras_matrix;
  vnl_diag_matrix_fixed<double, 3U> m_scale, m_lps_to_ras;
  vnl_vector_fixed<double, 3U> v_origin, v_ras_offset;

  // Compute the matrix
  m_dir = image->GetDirection().GetVnlMatrix();
  m_scale.set(image->GetSpacing().GetVnlVector());
  m_lps_to_ras.set(vnl_vector<double>(ImageType::ImageDimension, 1.0));
  m_lps_to_ras[0] = -1;
  m_lps_to_ras[1] = -1;
  m_ras_matrix = m_lps_to_ras * m_dir * m_scale;

  // Compute the vector
  v_origin = image->GetOrigin().GetVnlVector();
  v_ras_offset = m_lps_to_ras * v_origin;

  // Create the larger matrix
  TransformMatrixType mat;
  vnl_vector<double> vcol(ImageType::ImageDimension+1, 1.0);
  vcol.update(v_ras_offset);
  mat.SetIdentity();
  mat.GetVnlMatrix().update(m_ras_matrix);
  mat.GetVnlMatrix().set_column(ImageType::ImageDimension, vcol);

  return mat;
  }


template< class PixelType, unsigned int Dimension >
AntsTransform<itk::Transform<double,3,3>> fsl2antstransform( std::vector<std::vector<float> > matrix,
                AntsImage<itk::Image<PixelType, Dimension>> & ants_reference,
                AntsImage<itk::Image<PixelType, Dimension>> & ants_moving,
                int flag )
{
  typedef vnl_matrix_fixed<double, 4, 4>              MatrixType;
  typedef itk::Image<PixelType, Dimension>            ImageType;
  typedef itk::Matrix<double, 4,4>                    TransformMatrixType;

  typedef itk::AffineTransform<double, 3> AffTran;

  typedef typename ImageType::Pointer     ImagePointerType;

  typedef itk::Transform<double,3,3>                    TransformBaseType;
  typedef typename TransformBaseType::Pointer           TransformBasePointerType;

  ImagePointerType ref = ants_reference.ptr;
  ImagePointerType mov = ants_moving.ptr;

  MatrixType m_fsl, m_spcref, m_spcmov, m_swpref, m_swpmov, mat, m_ref, m_mov;

  //Rcpp::NumericMatrix matrix(r_matrix);
  for ( unsigned int i=0; i<matrix.size(); i++)
    for ( unsigned int j=0; j<matrix[0].size(); j++)
      m_fsl(i,j) = matrix[i][j];

  // Set the ref/mov matrices
  m_ref = GetVoxelSpaceToRASPhysicalSpaceMatrix<ImageType, TransformMatrixType>( ref ).GetVnlMatrix();
  m_mov = GetVoxelSpaceToRASPhysicalSpaceMatrix<ImageType, TransformMatrixType>( mov ).GetVnlMatrix();

  // Set the swap matrices
  m_swpref.set_identity();
  if(vnl_det(m_ref) > 0)
    {
    m_swpref(0,0) = -1.0;
    m_swpref(0,3) = (ref->GetBufferedRegion().GetSize(0) - 1) * ref->GetSpacing()[0];
    }

  m_swpmov.set_identity();
  if(vnl_det(m_mov) > 0)
    {
    m_swpmov(0,0) = -1.0;
    m_swpmov(0,3) = (mov->GetBufferedRegion().GetSize(0) - 1) * mov->GetSpacing()[0];
    }

  // Set the spacing matrices
  m_spcref.set_identity();
  m_spcmov.set_identity();
  for(size_t i = 0; i < 3; i++)
    {
    m_spcref(i,i) = ref->GetSpacing()[i];
    m_spcmov(i,i) = mov->GetSpacing()[i];
    }

  // Compute the output matrix
  //if (flag == FSL_TO_RAS)
    mat = m_mov * vnl_inverse(m_spcmov) * m_swpmov * vnl_inverse(m_fsl) * m_swpref * m_spcref * vnl_inverse(m_ref);

  // Add access to this
  // NOTE: m_fsl is really m_ras here
  //if (flag == RAS_TO_FSL)
  //  mat =
  //   vnl_inverse(vnl_inverse(m_swpmov) * m_spcmov* vnl_inverse(m_mov) *
  //  m_fsl *
  //  m_ref*vnl_inverse(m_spcref)*vnl_inverse(m_swpref));

  ///////////////

  // Flip the entries that must be flipped
  mat(2,0) *= -1; mat(2,1) *= -1;
  mat(0,2) *= -1; mat(1,2) *= -1;
  mat(0,3) *= -1; mat(1,3) *= -1;

  // Create an ITK affine transform
  AffTran::Pointer atran = AffTran::New();

  // Populate its matrix
  AffTran::MatrixType amat = atran->GetMatrix();
  AffTran::OffsetType aoff = atran->GetOffset();

  for(size_t r = 0; r < 3; r++)
    {
    for(size_t c = 0; c < 3; c++)
      {
      amat(r,c) = mat(r,c);
      }
    aoff[r] = mat(r,3);
    }

  atran->SetMatrix(amat);
  atran->SetOffset(aoff);

  TransformBasePointerType itkTransform = dynamic_cast<TransformBaseType*>( atran.GetPointer() );

  AntsTransform<TransformBaseType> out_ants_tx = { itkTransform };
  return out_ants_tx;
}


void local_fsl2antstransform(nb::module_ &m)
{
  m.def("fsl2antstransformF3", &fsl2antstransform<float,3>);
}
