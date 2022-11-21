#!/bin/bash

# ---------------------------------------------
# clone pybind11 into library directory

cd ants/lib # go to lib dir
git clone https://github.com/stnava/pybind11.git
cd ../../ # go back to main dir

# ---------------------------------------------
# create local ~/.antspy dir and move package data to it

mkdir -p ~/.antspy
cp data/* ~/.antspy/

# ---------------------------------------------
# clone ANTs and move all files into library directory

antsgit=https://github.com/ANTsX/ANTs.git
antstag=8cb575a08aaab579934ca181134db97cd069fe74 #
echo "ANTS;${antstag}" >> ./data/softwareVersions.csv

cd ants/lib # go to lib dir

# if antscore doesnt exist, create it
git clone $antsbranch $antsgit antsrepo

mkdir -p antscore

cd antsrepo # go to antscore
# check out right branch
git checkout master
git pull
git checkout $antstag

cd ..
# copy antscore files into library
cp -r antsrepo/Examples/*  antscore/
cp -r antsrepo/Examples/include/*  antscore
cp -r antsrepo/ImageRegistration/*  antscore/
cp -r antsrepo/ImageSegmentation/*  antscore/
cp -r antsrepo/Tensor/*  antscore/
cp -r antsrepo/Temporary/*  antscore/
cp -r antsrepo/Utilities/*  antscore/


rm -rf antsrepo # remove directory

# lil hack bc of stupid angled import bug in actual files
cp ReadWriteData.h antscore/ReadWriteData.h
# lil hack bc ANTsVersionConfig.h is only created if you build ANTs...
cp ANTsVersionConfig.h antscore/ANTsVersionConfig.h

cd  ../../ # go back to main dir
