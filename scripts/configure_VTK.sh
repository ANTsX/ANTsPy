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
vtkgit=https://github.com/Kitware/VTK.git
vtktag=acc5f269186e3571fb2a10af4448076ecac75e8e

if [[ -d vtksource ]]; then 
    if [[ ! -d vtksource/.git ]]; then
        rm -rf vtksource/
    fi  
fi
# if no directory, clone ITK in `itksource` dir
if [[ ! -d vtksource ]]; then 
    git clone $vtkgit vtksource 
fi
cd vtksource
if [[ -d .git ]]; then
    git checkout master;
    git checkout $vtktag
    rm -rf .git    
fi
# go back to main dir
cd ../
#if [[ ! -d ../data/ ]] ; then
#  mkdir -p ../data
#fi

echo "VTK;${vtktag}" >> ./data/softwareVersions.csv

mkdir vtkbuild
cd vtkbuild
cmake \
    -DCMAKE_BUILD_TYPE:STRING="${CMAKE_BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -Wno-c++11-long-long -fPIC -O2 -DNDEBUG  "\
    -DBUILD_SHARED_LIBS:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DVTK_LEGACY_REMOVE:BOOL=OFF \
    -DVTK_WRAP_PYTHON:BOOL=OFF ../vtksource/
make -j ${j:-3}
#make install
cd ../
