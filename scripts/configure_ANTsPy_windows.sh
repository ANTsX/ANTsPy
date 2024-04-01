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
antstag=35d9381721b143c7bbd9d5f7f4ad853406351c1c # 04-01-2024
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
