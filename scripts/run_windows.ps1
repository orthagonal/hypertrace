$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Write-Host '1/3 Generate deterministic synthetic examples with Rust...'
    cargo run --release -- --out data/safety_v0.bin --seed 42 --train 6000 --val 1000 --test 1000 --long 1000
    if ($LASTEXITCODE -ne 0) { throw 'Rust generator failed' }

    Write-Host '2/3 Build GPU HDC kernels (RTX 2060 SUPER = sm_75)...'
    nvcc -O3 -std=c++17 -arch=sm_75 -I include cuda/hdc_safety.cu -o hdc_safety_gpu.exe
    if ($LASTEXITCODE -ne 0) { throw 'CUDA compilation failed' }

    Write-Host '3/3 Train and evaluate the GPU prototype...'
    .\hdc_safety_gpu.exe data/safety_v0.bin --verify
    if ($LASTEXITCODE -ne 0) { throw 'GPU evaluation failed' }
}
finally { Pop-Location }
