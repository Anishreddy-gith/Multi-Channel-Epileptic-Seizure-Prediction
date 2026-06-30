$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

Copy-Item -LiteralPath "main.pdf" -Destination "..\conference_paper.pdf" -Force
Write-Host "Built paper\latex\main.pdf"
Write-Host "Copied final PDF to paper\conference_paper.pdf"
