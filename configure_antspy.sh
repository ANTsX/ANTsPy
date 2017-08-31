#!/bin/bash

# clone pybind11
cd ants/lib
git clone https://github.com/ncullen93/pybind11.git
cd ../../

# create ~/.antspy dir and move data to it
if [[ -d ~/.antspy ]]; then 
    rm rf ~/.antspy
fi
mkdir ~/.antspy
cp data/* ~/.antspy/