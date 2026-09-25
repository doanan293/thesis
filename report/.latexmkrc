# Build settings for the paper. `latexmk` in this directory builds the PDF,
# `latexmk -pvc` rebuilds on every save and `latexmk -C` removes all outputs.
# LaTeX Workshop in VS Code picks this file up through its default latexmk recipe.
@default_files = ('SoICT2026_Agentic_RAG_Vietnamese_Pharmaceutical_Documents.tex');

$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1 %O %S';

# Run bibtex whenever the .bib changes, and let `latexmk -c` delete the .bbl too.
$bibtex_use = 2;
$clean_ext = 'synctex.gz synctex(busy) run.xml';

# The paper must build without warnings: undefined citations or references,
# and any LaTeX, class or package warning, fail the build.
$warnings_as_errors = 1;
