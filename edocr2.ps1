# eDOCr2 launcher for Windows.
#
#   .\edocr2.cmd setup     make .venv (Python 3.11, PyTorch for the GPU if there is one), then fetch
#                          the models and Tesseract languages
#   .\edocr2.cmd doctor    report what was found
#   .\edocr2.cmd run IN... --out DIR
#
# The environment is made with uv, which links packages from its cache, so a PyTorch build that
# another project on this machine already uses costs no new download and almost no disk.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$torch = 'torch==2.11.0', 'torchvision==0.26.0'

function Find-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) { return $uv.Source }
    foreach ($p in "$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe") { if (Test-Path $p) { return $p } }
    throw 'uv not found. Install it (https://docs.astral.sh/uv/) or put it on PATH.'
}

if ($args.Count -gt 0 -and $args[0] -eq 'setup') {
    $uv = Find-Uv
    if (-not (Test-Path $python)) {
        & $uv venv --python 3.11 $venv
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
    # CUDA 12.8 builds run on every NVIDIA GPU from the GTX 900 series to the RTX 50 series
    $gpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $index = if ($gpu) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
    & $uv pip install --python $python @torch --index-url $index
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $uv pip install --python $python -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

if (-not (Test-Path $python)) {
    Write-Host 'No environment yet: run  .\edocr2.cmd setup'
    exit 1
}
$env:PYTHONPATH = $root
& $python -m edocr2 @args
exit $LASTEXITCODE
