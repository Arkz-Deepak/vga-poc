# Figures Directory

This directory is designed to hold PDF, PNG, or vector graphics for the VGA paper.

- If you place a vector diagram named `vga_architecture.pdf` in this directory, `main.tex` and `sections/03_Method.tex` will automatically detect it and render it as Fig. 1.
- If `vga_architecture.pdf` is not provided, the paper automatically falls back to the native, self-contained TikZ block diagram of the VGA architecture, ensuring zero compilation errors on Overleaf or local TeX engines.
