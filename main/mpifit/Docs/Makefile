# Build DDASToys documentation.

all: doxygen docbook

docbook:
	docbook2html -o ./manual manual.xml
	docbook2pdf -o ./manual manual.xml

doxygen:
	doxygen Doxyfile

clean:
	rm -rf sourcedocs manual
