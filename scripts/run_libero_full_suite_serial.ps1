param(
    [string]$Python = "",
    [string]$LiberoSource = "",
    [string]$LiberoConfig = "",
    [int]$StopAfterTasks = 0,
    [switch]$IncludeLibero90,
    [double]$MinAvailableRamGb = 1.5,
    [double]$MinDiskFreeGb = 2.0,
    [double]$SleepBetweenTasks = 5.0
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $Python) {
    if ($env:LIBERO_PYTHON) {
        $Python = $env:LIBERO_PYTHON
    } else {
        $Python = Join-Path (Split-Path $Repo -Parent) "external_benchmarks\.venvs\libero310\Scripts\python.exe"
    }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "LIBERO Python interpreter not found: $Python"
}

if (-not $LiberoSource) {
    if ($env:LIBERO_SOURCE_PATH) {
        $LiberoSource = $env:LIBERO_SOURCE_PATH
    } else {
        $LiberoSource = Join-Path (Split-Path $Repo -Parent) "external_benchmarks\LIBERO"
    }
}
if (Test-Path -LiteralPath $LiberoSource) {
    $env:LIBERO_SOURCE_PATH = $LiberoSource
}

if (-not $LiberoConfig) {
    if ($env:LIBERO_CONFIG_PATH) {
        $LiberoConfig = $env:LIBERO_CONFIG_PATH
    } else {
        $LiberoConfig = Join-Path (Split-Path $Repo -Parent) "external_benchmarks\.libero"
    }
}
if (Test-Path -LiteralPath $LiberoConfig) {
    $env:LIBERO_CONFIG_PATH = $LiberoConfig
}

$env:PYTHONPATH = "$Repo\src;$Repo\experiments;$LiberoSource;$env:PYTHONPATH"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

$ArgsList = @(
    "experiments/benchmark_libero_full_suite_serial.py",
    "--min-available-ram-gb", "$MinAvailableRamGb",
    "--min-disk-free-gb", "$MinDiskFreeGb",
    "--sleep-between-tasks", "$SleepBetweenTasks",
    "--low-priority"
)
if ($StopAfterTasks -gt 0) {
    $ArgsList += @("--stop-after-tasks", "$StopAfterTasks")
}
if ($IncludeLibero90) {
    $ArgsList += "--include-libero-90"
}

Push-Location $Repo
try {
    & $Python @ArgsList
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
