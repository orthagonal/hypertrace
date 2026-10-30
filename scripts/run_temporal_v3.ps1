param(
    [string]$OutputDirectory = 'target/temporal_v3',
    [ValidateRange(1, 1000)][int]$Repeats = 15,
    [string]$Cxx = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
$windowSavedPath = $env:PATH
try {
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $windowBuild = (Resolve-Path -LiteralPath $OutputDirectory).Path
    if (-not $Cxx) {
        if (Get-Command cl.exe -ErrorAction SilentlyContinue) { $Cxx = 'cl.exe' }
        elseif (Get-Command g++.exe -ErrorAction SilentlyContinue) { $Cxx = 'g++.exe' }
        else { throw 'Use a CUDA/MSVC developer shell or provide a configured C++17 compiler with -Cxx.' }
    }
    if (Test-Path -LiteralPath $Cxx) { $env:PATH = (Split-Path -Parent (Resolve-Path -LiteralPath $Cxx).Path) + ';' + $env:PATH }
    function Invoke-WindowTool([string]$Executable, [string[]]$ToolArguments) {
        & $Executable @ToolArguments
        if ($LASTEXITCODE -ne 0) { throw "Windowed command failed: $Executable (exit $LASTEXITCODE)" }
    }
    function Build-WindowCpp([string]$Source, [string]$Name) {
        $windowExe = Join-Path $windowBuild "$Name.exe"
        if ((Split-Path -Leaf $Cxx) -match '^cl(\.exe)?$') {
            Invoke-WindowTool $Cxx @('/nologo', '/O2', '/EHsc', '/std:c++17', '/W4',
                "/Fe$windowExe", "/Fo$(Join-Path $windowBuild "$Name.obj")", $Source)
        } else {
            Invoke-WindowTool $Cxx @('-O3', '-std=c++17', '-Wall', '-Wextra', '-pedantic', $Source, '-o', $windowExe)
        }
    }
    $windowSnapshot = Join-Path $windowBuild 'immutable_before.json'
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/verify_immutable.py', 'record', $windowSnapshot)
    $windowRegression = Join-Path $windowBuild 'regressions'
    & (Join-Path $PSScriptRoot 'run_temporal_v2.ps1') -OutputDirectory $windowRegression -Repeats 1 -Cxx $Cxx
    Build-WindowCpp 'experiments/temporal_v3/cpu.cpp' 'temporal_v3_cpu'
    Build-WindowCpp 'experiments/temporal_v3/tests.cpp' 'temporal_v3_tests'
    Invoke-WindowTool 'nvcc' @('-O3', '-std=c++17', '-arch=sm_75', 'experiments/temporal_v3/gpu.cu',
        '-o', (Join-Path $windowBuild 'temporal_v3_gpu.exe'))
    Invoke-WindowTool (Join-Path $windowBuild 'temporal_v3_tests.exe') @()
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/test_metrics.py')
    $windowV2Cpu = Join-Path $windowRegression 'temporal_v2_cpu.exe'
    $windowV2Gpu = Join-Path $windowRegression 'temporal_v2_gpu.exe'
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/test_integration.py', '--cpu', (Join-Path $windowBuild 'temporal_v3_cpu.exe'),
        '--gpu', (Join-Path $windowBuild 'temporal_v3_gpu.exe'), '--v2-cpu', $windowV2Cpu)
    $windowValidation = Join-Path $windowBuild 'validation'
    $windowTest = Join-Path $windowBuild 'test'
    $windowFrozen = Join-Path $windowBuild 'frozen_settings.json'
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/generate.py', '--out', (Join-Path $windowValidation 'data'), '--phase', 'validation')
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v2/run_collections.py', '--data', (Join-Path $windowValidation 'data'),
        '--out', (Join-Path $windowValidation 'anchors'), '--cpu', $windowV2Cpu, '--gpu', $windowV2Gpu, '--repeats', '1')
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/run_collections.py', '--data', (Join-Path $windowValidation 'data'),
        '--anchors', (Join-Path $windowValidation 'anchors'), '--out', (Join-Path $windowValidation 'results'),
        '--cpu', (Join-Path $windowBuild 'temporal_v3_cpu.exe'), '--gpu', (Join-Path $windowBuild 'temporal_v3_gpu.exe'), '--repeats', "$Repeats")
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/selection.py', '--data', (Join-Path $windowValidation 'data'),
        '--results', (Join-Path $windowValidation 'results'), '--out', $windowFrozen)
    # Test data/retrieval cannot run without the validation-bound code/settings lock.
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/generate.py', '--out', (Join-Path $windowTest 'data'), '--phase', 'test', '--frozen', $windowFrozen)
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v2/run_collections.py', '--data', (Join-Path $windowTest 'data'),
        '--out', (Join-Path $windowTest 'anchors'), '--cpu', $windowV2Cpu, '--gpu', $windowV2Gpu, '--repeats', '1')
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/run_collections.py', '--data', (Join-Path $windowTest 'data'),
        '--anchors', (Join-Path $windowTest 'anchors'), '--out', (Join-Path $windowTest 'results'),
        '--cpu', (Join-Path $windowBuild 'temporal_v3_cpu.exe'), '--gpu', (Join-Path $windowBuild 'temporal_v3_gpu.exe'),
        '--repeats', "$Repeats", '--frozen', $windowFrozen)
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/make_report.py', '--data', (Join-Path $windowTest 'data'),
        '--anchors', (Join-Path $windowTest 'anchors'), '--results', (Join-Path $windowTest 'results'), '--frozen', $windowFrozen,
        '--out', (Join-Path $windowBuild 'REPORT.md'))
    Invoke-WindowTool 'python' @('-B', 'experiments/temporal_v3/verify_immutable.py', 'verify', $windowSnapshot)
} finally {
    $env:PATH = $windowSavedPath
    Pop-Location
}
