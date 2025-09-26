#include "readImageMetadata.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "itkImageIOBase.h"
#include "itkImageIOFactory.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkMatrix.h"
#include "itkVersion.h"

namespace nb = nanobind;
using namespace nb::literals;

// Convert a dictionary entry to a Python object:
// - strings stay strings
// - numerics become Python numbers
// - itk::Matrix<float/double, 3/4, 3/4> -> list-of-lists
static nb::object MetaValueToPython(const itk::MetaDataDictionary &dict,
                                    const std::string &key) {
  // 1) strings
  {
    std::string s;
    if (itk::ExposeMetaData<std::string>(dict, key, s))
      return nb::cast(s);
  }

  // 2) numerics
  #define TRY_NUM(T) do { T v{}; if (itk::ExposeMetaData<T>(dict, key, v)) return nb::cast(v); } while (0)
  TRY_NUM(double);
  TRY_NUM(float);
  TRY_NUM(long double);
  TRY_NUM(long long);
  TRY_NUM(unsigned long long);
  TRY_NUM(long);
  TRY_NUM(unsigned long);
  TRY_NUM(int);
  TRY_NUM(unsigned int);
  TRY_NUM(short);
  TRY_NUM(unsigned short);
  TRY_NUM(char);
  TRY_NUM(unsigned char);
  TRY_NUM(bool);
  #undef TRY_NUM

  // 3) matrices (qto_xyz/sto_xyz may be stored as itk::Matrix)
  auto matrix_to_list = [](auto &&M) {
    using Mat = std::decay_t<decltype(M)>;
    constexpr unsigned R = Mat::RowDimensions;
    constexpr unsigned C = Mat::ColumnDimensions;
    nb::list rows;
    for (unsigned r = 0; r < R; ++r) {
      nb::list row;
      for (unsigned c = 0; c < C; ++c)
        row.append(M(r, c));
      rows.append(row);
    }
    return rows;
  };

  #define TRY_MAT(T,R,C) do { \
    itk::Matrix<T,R,C> M; \
    if (itk::ExposeMetaData<itk::Matrix<T,R,C>>(dict, key, M)) \
      return matrix_to_list(M); \
  } while (0)

  TRY_MAT(double,4,4);  TRY_MAT(float,4,4);
  TRY_MAT(double,3,3);  TRY_MAT(float,3,3);
  #undef TRY_MAT

  // 4) fallback: textual Print() of the MetaDataObject
  auto it = dict.Find(key);
  if (it != dict.End()) {
    std::ostringstream oss;
    it->second->Print(oss);
    return nb::cast(oss.str());
  }
  return nb::none();
}

nb::dict readImageMetadata(const std::string &path) {
  itk::ImageIOBase::Pointer io =
      itk::ImageIOFactory::CreateImageIO(path.c_str(),
                                         itk::IOFileModeEnum::ReadMode);
  if (!io)
    throw std::runtime_error("Could not find an ImageIO for: " + path);

  io->SetFileName(path);
  io->ReadImageInformation(); // header only

  const auto &md = io->GetMetaDataDictionary();

  nb::dict out;

  auto keys = md.GetKeys();

  for (const auto &k : keys) {
    nb::object v = MetaValueToPython(md, k);
    if (!v.is_none())
      out[nb::cast(k)] = v;
  }

  return out;
}

void local_readImageMetadata(nb::module_ &m) {
  m.def("readImageMetadata", &readImageMetadata);
}
