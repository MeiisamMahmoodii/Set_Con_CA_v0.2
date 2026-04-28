# MiKTeX Build Script for Set-ConCA NeurIPS Manuscript
$Filename = "setconca_neurips"

# Ensure we are in the script's directory for pdflatex to find the .tex file
cd $PSScriptRoot

Write-Host "--- Phase 1: Initial PDFLaTeX ---"
pdflatex -interaction=nonstopmode $Filename

Write-Host "--- Phase 2: BibTeX ---"
bibtex $Filename

Write-Host "--- Phase 3: Resolving Citations (Run 1) ---"
pdflatex -interaction=nonstopmode $Filename

Write-Host "--- Phase 4: Resolving Citations (Run 2) ---"
pdflatex -interaction=nonstopmode $Filename

Write-Host "--- Compilation Complete: $Filename.pdf ---"
