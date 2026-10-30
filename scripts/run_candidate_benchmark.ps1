param(
    [string]$OutputDirectory = 'target/candidate_benchmark',
    [ValidateRange(1,100)][int]$Repeats = 5,
    [string]$Lengths = '32,1024,4096,32768',
    [string]$Galleries = '16,128,1024',
    [string]$VcVarsPath = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$candidateSavedPath = $env:PATH
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $candidateRtk = Get-Command rtk -ErrorAction SilentlyContinue
    function Invoke-CandidateTool([string]$Executable, [string[]]$ToolArguments) {
        if ($candidateRtk) { & $candidateRtk.Source proxy $Executable @ToolArguments }
        else { & $Executable @ToolArguments }
        if ($LASTEXITCODE -ne 0) { throw "Candidate benchmark command failed: $Executable (exit $LASTEXITCODE)" }
    }
    if (-not $candidateRtk) { Write-Host 'rtk is unavailable; running underlying commands.' }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        if (-not $VcVarsPath) {
            $candidateVswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
            if (Test-Path -LiteralPath $candidateVswhere) {
                $candidateInstall = & $candidateVswhere -latest -prerelease -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
                if ($candidateInstall) { $VcVarsPath = Join-Path $candidateInstall 'VC/Auxiliary/Build/vcvars64.bat' }
            }
        }
        if (-not $VcVarsPath -or -not (Test-Path -LiteralPath $VcVarsPath)) { throw 'MSVC not found. Use a developer PowerShell, or provide -VcVarsPath.' }
        # Import environment without opening another visible terminal.
        $candidateEnvironment = & cmd.exe /d /c "call `"$VcVarsPath`" >nul && set"
        if ($LASTEXITCODE -ne 0) { throw 'MSVC environment setup failed' }
        foreach ($candidateLine in $candidateEnvironment) {
            if ($candidateLine -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
        }
    }
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $candidateBuild = (Resolve-Path -LiteralPath $OutputDirectory).Path
    if (-not (Test-Path 'target/temporal_v3/test/data/manifest.json')) {
        throw 'Existing v3 dataset/results are required. Run scripts/run_temporal_v3.ps1 first; this benchmark never overwrites the original benchmark.'
    }
    function Build-CandidateCpp([string]$Source,[string]$Name) {
        Invoke-CandidateTool 'cl.exe' @('/nologo','/O2','/EHsc','/std:c++17','/W4','/Iinclude',
            "/Fe$(Join-Path $candidateBuild "$Name.exe")", "/Fo$(Join-Path $candidateBuild "$Name.obj")",$Source)
    }
    function Build-CandidateCuda([string]$Source,[string]$Name) {
        Invoke-CandidateTool 'nvcc' @('-O3','-std=c++17','-arch=sm_75','-Iinclude',$Source,'-o',(Join-Path $candidateBuild "$Name.exe"))
    }
    Write-Host 'Building candidate benchmark and all prior test executables...'
    Build-CandidateCuda 'experiments/candidate_benchmark/benchmark.cu' 'candidate_benchmark'
    Build-CandidateCuda 'experiments/candidate_benchmark/tests.cu' 'candidate_tests'
    Build-CandidateCpp 'tests/report_test.cpp' 'v0_report_tests'
    Build-CandidateCpp 'tests/diagnostics_test.cpp' 'v0_diagnostics_tests'
    Build-CandidateCpp 'cuda/hdc_reference.cpp' 'v0_cpu'
    Build-CandidateCuda 'cuda/hdc_safety.cu' 'v0_gpu'
    foreach ($candidateVersion in 1,2,3) {
        Build-CandidateCpp "experiments/temporal_v$candidateVersion/cpu.cpp" "v${candidateVersion}_cpu"
        Build-CandidateCpp "experiments/temporal_v$candidateVersion/tests.cpp" "v${candidateVersion}_tests"
        Build-CandidateCuda "experiments/temporal_v$candidateVersion/gpu.cu" "v${candidateVersion}_gpu"
    }
    $candidateLog = Join-Path $candidateBuild 'tests.log'
    Start-Transcript -Path $candidateLog -Force | Out-Null
    try {
        Write-Host 'Running v0/v1/v2/v3 and candidate regressions (no legacy benchmark reruns)...'
        Invoke-CandidateTool 'cargo' @('test')
        Invoke-CandidateTool 'cargo' @('build','--release')
        Invoke-CandidateTool (Join-Path $candidateBuild 'v0_report_tests.exe') @()
        Invoke-CandidateTool (Join-Path $candidateBuild 'v0_diagnostics_tests.exe') @()
        Invoke-CandidateTool 'python' @('-B','tests/diagnostics_integration_test.py','--cpu',(Join-Path $candidateBuild 'v0_cpu.exe'),'--gpu',(Join-Path $candidateBuild 'v0_gpu.exe'))
        Invoke-CandidateTool 'python' @('-B','tests/make_fixture.py','--out',(Join-Path $candidateBuild 'v0_fixture.bin'))
        Invoke-CandidateTool (Join-Path $candidateBuild 'v0_gpu.exe') @((Join-Path $candidateBuild 'v0_fixture.bin'),'--verify')
        if (Test-Path 'data/safety_v0.bin') { Invoke-CandidateTool (Join-Path $candidateBuild 'v0_gpu.exe') @('data/safety_v0.bin','--verify') }
        foreach ($candidateVersion in 1,2,3) { Invoke-CandidateTool (Join-Path $candidateBuild "v${candidateVersion}_tests.exe") @() }
        Invoke-CandidateTool 'python' @('-B','experiments/temporal_v1/test_integration.py','--generator',(Join-Path (Get-Location) 'target/release/hypercubedb-safety-sim.exe'),
            '--cpu',(Join-Path $candidateBuild 'v1_cpu.exe'),'--gpu',(Join-Path $candidateBuild 'v1_gpu.exe'))
        foreach ($candidateVersion in 2,3) { Invoke-CandidateTool 'python' @('-B',"experiments/temporal_v$candidateVersion/test_metrics.py") }
        Invoke-CandidateTool 'python' @('-B','experiments/temporal_v2/test_integration.py','--cpu',(Join-Path $candidateBuild 'v2_cpu.exe'),'--gpu',(Join-Path $candidateBuild 'v2_gpu.exe'))
        Invoke-CandidateTool 'python' @('-B','experiments/temporal_v3/test_integration.py','--cpu',(Join-Path $candidateBuild 'v3_cpu.exe'),'--gpu',(Join-Path $candidateBuild 'v3_gpu.exe'),'--v2-cpu',(Join-Path $candidateBuild 'v2_cpu.exe'))
        Invoke-CandidateTool (Join-Path $candidateBuild 'candidate_tests.exe') @()
        Invoke-CandidateTool 'python' @('-B','experiments/candidate_benchmark/test_pipeline.py','--exe',(Join-Path $candidateBuild 'candidate_benchmark.exe'),'--original','target/temporal_v3/test/data/dense_target_32')
    } finally { Stop-Transcript | Out-Null }
    Write-Host 'Running finite validation and held-out candidate suites; progress is printed per collection.'
    $candidateRunArgs = @('-B','experiments/candidate_benchmark/run.py','--exe',(Join-Path $candidateBuild 'candidate_benchmark.exe'),
        '--out',$candidateBuild,'--repeats',"$Repeats",'--lengths',$Lengths,'--galleries',$Galleries)
    if ($Resume) { $candidateRunArgs += '--resume' }
    Invoke-CandidateTool 'python' $candidateRunArgs
    $candidateCpuName = (Get-ItemProperty -LiteralPath 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0').ProcessorNameString
    $candidateCudaVersion = Invoke-CandidateTool 'nvcc' @('--version')
    Add-Content -LiteralPath (Join-Path $candidateBuild 'hardware.txt') -Value @(
        "CPU: $candidateCpuName", "MSVC tools: $env:VCToolsVersion",
        'Build flags: nvcc -O3 -std=c++17 -arch=sm_75; cl /O2 /EHsc /std:c++17',
        "Benchmark executable SHA256: $((Get-FileHash -LiteralPath (Join-Path $candidateBuild 'candidate_benchmark.exe') -Algorithm SHA256).Hash)",
        $candidateCudaVersion)
    Invoke-CandidateTool 'python' @('-B','experiments/candidate_benchmark/report.py','--results',$candidateBuild,'--out',(Join-Path $candidateBuild 'report'))
    Invoke-CandidateTool 'python' @('-B','experiments/candidate_benchmark/test_report.py',(Join-Path $candidateBuild 'report'))
    Write-Host "Complete. Read $(Join-Path $candidateBuild 'report/REPORT.md'). No changes were committed."
} finally {
    $env:PATH = $candidateSavedPath
    Pop-Location
}
