SET PYCMD=python
pushd "tests"
echo "Running core tests"
%PYCMD% test_core_ants_image.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
%PYCMD% test_core_ants_image_io.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
%PYCMD% test_core_ants_transform.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
%PYCMD% test_core_ants_transform_io.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
%PYCMD% test_core_ants_metric.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
echo "Running learn tests"
%PYCMD% test_learn.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
echo "Running registation tests"
%PYCMD% test_registation.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
echo "Running segmentation tests"
%PYCMD% test_segmentation.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
echo "Running utils tests"
%PYCMD% test_utils.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
echo "Running bug tests"
%PYCMD% test_bugs.py %@%
if ( $LastExitCode -ne 0 ) { exit 1 }
popd
