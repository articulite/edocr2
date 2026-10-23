# Make the drawgen sheets the benchmark and the training data use, from public parts only (the
# NIST AP242 test parts and drawgen's own fixtures), never a customer's.
#
#   benchmarks\make_drawgen_sets.ps1 -Drawgen <drawgen folder> -Out <data folder> [-TrainSeeds N]
#
# Parts are split, not seeds: the benchmark's eight parts never appear in training.
#   eval   CTC 01-05, FTC 06-07, STC 06; two seeds each from 101 up, a random style per
#          seed (16 sheets)
#   train  FTC 08-11, STC 07-10, the fixture and the holes fixture; seeds 1..N, random styles
# Needs drawgen's hole-callout notes (Aether-KK branch hole-callouts, record schema 13).
# One geometry process at a time, as drawgen's handoff asks.
param(
    [Parameter(Mandatory)][string]$Drawgen,
    [Parameter(Mandatory)][string]$Out,
    [int]$TrainSeeds = 25
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $Drawgen 'target\release\drawgen.exe'
if (-not $env:DRAWGEN_OCCT -and (Test-Path (Join-Path $Drawgen 'local\occt'))) { $env:DRAWGEN_OCCT = Join-Path $Drawgen 'local\occt' }
# The AP242 set drawgen's own acceptance runs use; the NIST folder holds other editions too.
$nist = Get-ChildItem (Join-Path $Drawgen 'local\nist\ap242') -Filter *.stp
$evalParts = 'ctc_01', 'ctc_02', 'ctc_03', 'ctc_04', 'ctc_05', 'ftc_06', 'ftc_07', 'stc_06'
$key = { param($f) $f.BaseName.Substring(5, 6) }   # nist_ctc_01_asme1_... -> ctc_01

function Make($part, $dir, $seed, $count) {
    & $exe make $part --out $dir --seed $seed --count $count --style random | Out-Null
    if ($LASTEXITCODE) { throw "drawgen make failed on $part" }
}

# Each eval part its own two seeds: drawgen makes some choices per seed (depth ranges,
# reference distances), and shared seeds would make the same choice on every sheet.
$i = 0
foreach ($p in $nist | Where-Object { $evalParts -contains (& $key $_) }) {
    Make $p.FullName (Join-Path $Out "eval\$(& $key $p)") (101 + 2 * $i) 2
    $i++
}
$fixtures = Join-Path $Out 'fixtures'
New-Item -ItemType Directory -Force $fixtures | Out-Null
& $exe fixture (Join-Path $fixtures 'fixture.step') | Out-Null
& $exe fixture (Join-Path $fixtures 'holes.step') --holes | Out-Null
$train = @($nist | Where-Object { $evalParts -notcontains (& $key $_) } | ForEach-Object { $_.FullName }) + @(Get-ChildItem $fixtures -Filter *.step | ForEach-Object { $_.FullName })
foreach ($p in $train) {
    Make $p (Join-Path $Out "train\$([IO.Path]::GetFileNameWithoutExtension($p))") 1 $TrainSeeds
}
$n = { param($d) (Get-ChildItem (Join-Path $Out $d) -Recurse -Filter record.json -ErrorAction SilentlyContinue).Count }
"eval sheets: $(& $n 'eval')  train sheets: $(& $n 'train')"
