#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}


if [[ $COVERAGE -eq 1 ]]; then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"


echo "Running core tests"
$PYCMD test_core.py $@

echo "Running utils tests"
$PYCMD test_core.py $@

popd
