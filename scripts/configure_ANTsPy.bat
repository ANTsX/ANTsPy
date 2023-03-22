:: Converted with the help of https://daniel-sc.github.io/bash-shell-to-bat-converter/
:: @echo off

:: clone pybind11 into library directory
cd ants\lib
if not exist %USERPROFILE%\pybind11\  (
  git clone https://github.com/stnava/pybind11.git
)
cd ..\..

echo %USERPROFILE%

:: create local ~/.antspy dir and move package data to it
if not exist %USERPROFILE%\.antspy\ (
  mkdir %USERPROFILE%\.antspy
)
COPY data\* %USERPROFILE%\.antspy

:: clone ANTs and move all files into library directory
SET antsgit=https://github.com/ANTsX/ANTs.git
SET antstag=276cf0717945d3dd3c4298c607d9d6a788ba574e
echo "ANTS;%antstag%" REM UNKNOWN: {"type":"Redirect","op":{"text":">>","type":"dgreat"},"file":{"text":"./data/softwareVersions.csv","type":"Word"}}
cd ants\lib
echo "123"
:: if antscore doesnt exist, create it
if not exist "antscore\" (
  echo "2"
  git clone %antsgit% antsrepo
  mkdir antscore
  cd antsrepo

  :: check out right branch
  if exist ".git\" (
    git checkout master
    git pull
    git checkout %antstag%
  )
  cd ..
  COPY  antsrepo\Examples\* antscore\
  COPY  antsrepo\Examples\include\* antscore
  XCOPY  antsrepo\Examples\include\ antscore\include\
  COPY  antsrepo\ImageRegistration\* antscore\
  COPY  antsrepo\ImageSegmentation\* antscore\
  COPY  antsrepo\Tensor\* antscore\
  COPY  antsrepo\Temporary\* antscore\
  COPY  antsrepo\Utilities\* antscore\
  echo "HERE"

  :: lil hack bc of stupid angled import bug in actual files
  :: cp ReadWriteData.h antscore/ReadWriteData.h
  :: lil hack bc ANTsVersionConfig.h is only created if you build ANTs...
  COPY  ANTsVersionConfig.h antscore\ANTsVersionConfig.h
)
cd ..\..
