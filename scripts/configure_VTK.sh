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
# vtktag=c5138560409c75408ff76bccff938f21e5dcafc6 #4.12

# if there is a directory but no git,
# remove it
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

mkdir -p vtkbuild
cd vtkbuild
ccmake ../vtksource/
make -j 2
#make install
cd ../


