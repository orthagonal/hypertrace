param(
    [string]$OutputDirectory = 'target/temporal_v1',
    [ValidateRange(1, 1000)][int]$Repeats = 15,
    [string]$Cxx = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
$temporalSavedPath = $env:PATH
try {
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $temporalBuild = (Resolve-Path -LiteralPath $OutputDirectory).Path
    if (-not $Cxx) {
        if (Get-Command cl.exe -ErrorAction SilentlyContinue) { $Cxx = 'cl.exe' }
        elseif (Get-Command g++.exe -ErrorAction SilentlyContinue) { $Cxx = 'g++.exe' }
        else { throw 'Use a CUDA/MSVC developer shell or supply -Cxx with a configured C++17 compiler.' }
    }
    if (Test-Path -LiteralPath $Cxx) { $env:PATH = (Split-Path -Parent (Resolve-Path -LiteralPath $Cxx).Path) + ';' + $env:PATH }
    function Invoke-TemporalTool([string]$Executable, [string[]]$ToolArguments) {
        & $Executable @ToolArguments
        if ($LASTEXITCODE -ne 0) { throw "Temporal command failed: $Executable (exit $LASTEXITCODE)" }
    }
    function Build-TemporalCpp([string]$Source, [string]$Name) {
        $temporalExe = Join-Path $temporalBuild "$Name.exe"
        if ((Split-Path -Leaf $Cxx) -match '^cl(\.exe)?$') {
            Invoke-TemporalTool $Cxx @('/nologo', '/O2', '/EHsc', '/std:c++17', '/W4',
                "/Fe$temporalExe", "/Fo$(Join-Path $temporalBuild "$Name.obj")", $Source)
        } else {
            Invoke-TemporalTool $Cxx @('-O3', '-std=c++17', '-Wall', '-Wextra', '-pedantic', $Source, '-o', $temporalExe)
        }
    }
    Invoke-TemporalTool 'cargo' @('test')
    Invoke-TemporalTool 'cargo' @('build', '--release')
    Build-TemporalCpp 'experiments/temporal_v1/cpu.cpp' 'temporal_cpu'
    Build-TemporalCpp 'experiments/temporal_v1/tests.cpp' 'temporal_tests'
    Invoke-TemporalTool 'nvcc' @('-O3', '-std=c++17', '-arch=sm_75', 'experiments/temporal_v1/gpu.cu',
        '-o', (Join-Path $temporalBuild 'temporal_gpu.exe'))
    Invoke-TemporalTool (Join-Path $temporalBuild 'temporal_tests.exe') @()
    $temporalGenerator = (Resolve-Path 'target/release/hypercubedb-safety-sim.exe').Path
    Invoke-TemporalTool 'python' @('experiments/temporal_v1/test_integration.py', '--generator', $temporalGenerator,
        '--cpu', (Join-Path $temporalBuild 'temporal_cpu.exe'), '--gpu', (Join-Path $temporalBuild 'temporal_gpu.exe'))
    $temporalData = Join-Path $temporalBuild 'events.bin'
    Invoke-TemporalTool $temporalGenerator @('--temporal-v1', '--out', $temporalData, '--seed', '42')
    Invoke-TemporalTool (Join-Path $temporalBuild 'temporal_cpu.exe') @($temporalData, '--out', (Join-Path $temporalBuild 'cpu'))
    Invoke-TemporalTool (Join-Path $temporalBuild 'temporal_gpu.exe') @($temporalData, '--out', (Join-Path $temporalBuild 'gpu'), '--repeats', "$Repeats")
    Invoke-TemporalTool 'python' @('experiments/temporal_v1/make_report.py', '--dataset', $temporalData,
        '--results', (Join-Path $temporalBuild 'gpu'), '--cpu-results', (Join-Path $temporalBuild 'cpu'),
        '--out', (Join-Path $temporalBuild 'REPORT.md'))
} finally {
    $env:PATH = $temporalSavedPath
    Pop-Location
}
