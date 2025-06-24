$PYCMD = "python"

$PYTHON_SCRIPTS = @(
    "test_core_ants_image.py",
    "test_core_ants_image_io.py",
    "test_core_ants_transform.py",
    "test_core_ants_transform_io.py",
    "test_core_ants_metric.py",
    "test_learn.py",
    "test_registration.py",
    "test_segmentation.py",
    "test_utils.py",
    "test_bugs.py",
    "test_deeplearn.py"
)

$scriptFailed = $false

Set-Location -Path ".\tests"

foreach ($script in $PYTHON_SCRIPTS) {
    Write-Host "Running Python script $script"
    & $PYCMD $script $args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python script $script failed with exit code $LASTEXITCODE"
        $scriptFailed = $true
    }
}

if ($scriptFailed) {
    exit 1
}
