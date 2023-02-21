# Build DDASToys documentation.

XSLTPROC = /usr/bin/xsltproc
XSL = /usr/share/xml/docbook/stylesheet/nwalsh/xhtml/docbook.xsl

all: UserGuide.html doxygen

UserGuide.html: UserGuide.xml $(XSL)
	@$(XSLTPROC) -o $@ $(XSL) $<

doxygen:
	doxygen Doxyfile

clean:
	rm -f UserGuide.html
	rm -rf html latex
