#!/bin/bash
CXX_STD=CXX11
JTHREADS=2
if [[ `uname` -eq Darwin ]] ; then
  CMAKE_BUILD_TYPE=Release
fi
if [[ $TRAVIS -eq true ]] ; then
  CMAKE_BUILD_TYPE=Release
  JTHREADS=2
fi

#cd ./src
itkgit=https://github.com/InsightSoftwareConsortium/ITK.git
itktag=81c14ce858a530699ee2fbf7fa48b884ad26b984
# if there is a directory but no git, remove it
if [[ -d itksource ]]; then
    if [[ ! -d itksource/.git ]]; then
        rm -rf itksource/
    fi
fi
# if no directory, clone ITK in `itksource` dir
if [[ ! -d itksource ]]; then
    git clone $itkgit itksource
fi
cd itksource
if [[ -d .git ]]; then
    git checkout master;
    git checkout $itktag
    rm -rf .git
fi
# go back to main dir
cd ../
#if [[ ! -d ../data/ ]] ; then
#  mkdir -p ../data
#fi

echo "Dependency;GitTag" > ./data/softwareVersions.csv
echo "ITK;${itktag}" >> ./data/softwareVersions.csv

mkdir -p itkbuild
cd itkbuild
compflags=" -fPIC -O2  "
cmake \
    -DCMAKE_BUILD_TYPE:STRING="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DITK_USE_GIT_PROTOCOL:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH=${R_PACKAGE_DIR}/libs/  \
    -DITK_LEGACY_REMOVE:BOOL=OFF  \
    -DITK_FUTURE_LEGACY_REMOVE:=BOOL=ON \
    -DITK_BUILD_DEFAULT_MODULES:BOOL=OFF \
    -DKWSYS_USE_MD5:BOOL=ON \
    -DITK_WRAPPING:BOOL=OFF \
    -DModule_MGHIO:BOOL=ON \
    -DModule_ITKDeprecated:BOOL=OFF \
    -DModule_ITKReview:BOOL=ON \
    -DModule_ITKVtkGlue:BOOL=OFF \
    -DITKGroup_Core=ON \
    -DModule_ITKReview=ON \
    -DITKGroup_Filtering=ON \
    -DITKGroup_IO=ON \
    -DITKGroup_Numerics=ON \
    -DITKGroup_Registration=ON \
    -DITKGroup_Segmentation=ON \
    -DModule_GenericLabelInterpolator:BOOL=ON \
    -DCMAKE_C_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_CXX_VISIBILITY_PRESET:BOOL=hidden \
    -DCMAKE_VISIBILITY_INLINES_HIDDEN:BOOL=ON ../itksource/
make -j 4
#make install
cd ../
