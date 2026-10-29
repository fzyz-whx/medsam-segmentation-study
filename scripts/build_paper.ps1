# Build from paper/ so bibliography and figure paths resolve correctly.
# Publish only the current paper/main.pdf.
$ErrorActionPreference = 'Stop'
$rootDir = Split-Path -Parent $PSScriptRoot
$paperDir = Join-Path $rootDir 'paper'
$buildDir = Join-Path $rootDir 'logs/latex-review'
$oldLocation = Get-Location
$oldBibInputs = $env:BIBINPUTS
$oldBstInputs = $env:BSTINPUTS
try {
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    Set-Location -LiteralPath $paperDir
    & xelatex '-interaction=nonstopmode' '-halt-on-error' '-jobname=main-review' "-output-directory=$buildDir" 'main.tex'
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX pass 1 failed.' }
    $env:BIBINPUTS = "$paperDir;$buildDir;$oldBibInputs;"
    $env:BSTINPUTS = "$paperDir;$oldBstInputs;"
    Set-Location -LiteralPath $buildDir
    & bibtex 'main-review'
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    Set-Location -LiteralPath $paperDir
    foreach ($pass in 2..3) {
        & xelatex '-interaction=nonstopmode' '-halt-on-error' '-jobname=main-review' "-output-directory=$buildDir" 'main.tex'
        if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed." }
    }
    $log = Get-Content -LiteralPath (Join-Path $buildDir 'main-review.log') -Raw
    if ($log -match '(Citation|Reference)[^\r\n]*undefined|There were undefined|Missing character|Overfull \\[hv]box') {
        throw 'Unresolved reference, missing glyph, or overfull box; inspect logs/latex-review/main-review.log.'
    }
    Copy-Item -LiteralPath (Join-Path $buildDir 'main-review.pdf') -Destination (Join-Path $paperDir 'main.pdf')
    Write-Host 'Paper PDF written: paper/main.pdf'
}
finally {
    $env:BIBINPUTS = $oldBibInputs
    $env:BSTINPUTS = $oldBstInputs
    Set-Location -LiteralPath $oldLocation
}
