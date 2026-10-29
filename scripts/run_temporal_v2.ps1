param(
    [string]$OutputDirectory = 'target/temporal_v2',
    [ValidateRange(1, 1000)][int]$Repeats = 15,
    [string]$Cxx = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
$sparseSavedPath = $env:PATH
try {
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $sparseBuild = (Resolve-Path -LiteralPath $OutputDirectory).Path
    if (-not $Cxx) {
        if (Get-Command cl.exe -ErrorAction SilentlyContinue) { $Cxx = 'cl.exe' }
        elseif (Get-Command g++.exe -ErrorAction SilentlyContinue) { $Cxx = 'g++.exe' }
        else { throw 'Use a CUDA/MSVC developer shell or supply -Cxx with a configured C++17 compiler.' }
    }
    if (Test-Path -LiteralPath $Cxx) { $env:PATH = (Split-Path -Parent (Resolve-Path -LiteralPath $Cxx).Path) + ';' + $env:PATH }
    function Invoke-SparseTool([string]$Executable, [string[]]$ToolArguments) {
        & $Executable @ToolArguments
        if ($LASTEXITCODE -ne 0) { throw "Sparse command failed: $Executable (exit $LASTEXITCODE)" }
    }
    function Build-SparseCpp([string]$Source, [string]$Name) {
        $sparseExe = Join-Path $sparseBuild "$Name.exe"
        if ((Split-Path -Leaf $Cxx) -match '^cl(\.exe)?$') {
            Invoke-SparseTool $Cxx @('/nologo', '/O2', '/EHsc', '/std:c++17', '/W4', '/Iinclude',
                "/Fe$sparseExe", "/Fo$(Join-Path $sparseBuild "$Name.obj")", $Source)
        } else {
            Invoke-SparseTool $Cxx @('-O3', '-std=c++17', '-Wall', '-Wextra', '-pedantic', '-I', 'include', $Source, '-o', $sparseExe)
        }
    }
    $sparseSnapshot = Join-Path $sparseBuild 'immutable_before.json'
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/verify_immutable.py', 'record', $sparseSnapshot)
    # Preserve and exercise all previous tests through their unchanged workflow.
    & (Join-Path $PSScriptRoot 'run_temporal_v1.ps1') -OutputDirectory (Join-Path $sparseBuild 'regression_v1') -Repeats 1 -Cxx $Cxx
    Build-SparseCpp 'tests/report_test.cpp' 'regression_report_test'
    Build-SparseCpp 'tests/diagnostics_test.cpp' 'regression_diagnostics_test'
    Build-SparseCpp 'cuda/hdc_reference.cpp' 'regression_v0_cpu'
    Invoke-SparseTool 'nvcc' @('-O3', '-std=c++17', '-arch=sm_75', '-I', 'include', 'cuda/hdc_safety.cu',
        '-o', (Join-Path $sparseBuild 'regression_v0_gpu.exe'))
    Invoke-SparseTool (Join-Path $sparseBuild 'regression_report_test.exe') @()
    Invoke-SparseTool (Join-Path $sparseBuild 'regression_diagnostics_test.exe') @()
    Invoke-SparseTool 'python' @('-B', 'tests/diagnostics_integration_test.py',
        '--cpu', (Join-Path $sparseBuild 'regression_v0_cpu.exe'), '--gpu', (Join-Path $sparseBuild 'regression_v0_gpu.exe'))
    $sparseFixture = Join-Path $sparseBuild 'regression_fixture.bin'
    Invoke-SparseTool 'python' @('-B', 'tests/make_fixture.py', '--out', $sparseFixture)
    Invoke-SparseTool (Join-Path $sparseBuild 'regression_v0_gpu.exe') @($sparseFixture, '--verify')
    if (Test-Path -LiteralPath 'data/safety_v0.bin') {
        Invoke-SparseTool (Join-Path $sparseBuild 'regression_v0_gpu.exe') @('data/safety_v0.bin', '--verify')
    }
    Build-SparseCpp 'experiments/temporal_v2/cpu.cpp' 'temporal_v2_cpu'
    Build-SparseCpp 'experiments/temporal_v2/tests.cpp' 'temporal_v2_tests'
    Invoke-SparseTool 'nvcc' @('-O3', '-std=c++17', '-arch=sm_75', 'experiments/temporal_v2/gpu.cu',
        '-o', (Join-Path $sparseBuild 'temporal_v2_gpu.exe'))
    Invoke-SparseTool (Join-Path $sparseBuild 'temporal_v2_tests.exe') @()
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/test_metrics.py')
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/test_integration.py',
        '--cpu', (Join-Path $sparseBuild 'temporal_v2_cpu.exe'), '--gpu', (Join-Path $sparseBuild 'temporal_v2_gpu.exe'))
    $sparseData = Join-Path $sparseBuild 'data'
    $sparseResults = Join-Path $sparseBuild 'results'
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/generate.py', '--out', $sparseData)
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/run_collections.py', '--data', $sparseData,
        '--out', $sparseResults, '--cpu', (Join-Path $sparseBuild 'temporal_v2_cpu.exe'),
        '--gpu', (Join-Path $sparseBuild 'temporal_v2_gpu.exe'), '--repeats', "$Repeats")
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/make_report.py', '--data', $sparseData,
        '--results', $sparseResults, '--out', (Join-Path $sparseBuild 'REPORT.md'))
    Invoke-SparseTool 'python' @('-B', 'experiments/temporal_v2/verify_immutable.py', 'verify', $sparseSnapshot)
} finally {
    $env:PATH = $sparseSavedPath
    Pop-Location
}
