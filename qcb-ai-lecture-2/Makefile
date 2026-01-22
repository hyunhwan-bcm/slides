.PHONY: all clean view

all: slides.pdf

slides.pdf: slides.tex
	pdflatex -shell-escape slides.tex
	pdflatex -shell-escape slides.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb _minted-*

view: slides.pdf
	open slides.pdf

help:
	@echo "Available targets:"
	@echo "  make          - Build the PDF"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make view     - Build and open PDF"
