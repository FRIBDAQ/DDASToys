# SPECTCLHOME must be define externally to point to a SpecTcl that has
# the fragment index stuff built in.

SPECINC=$(SPECTCLHOME)/include
SPECLIB=$(SPECTCLHOME)/lib

CXXFLAGS=-I$(DAQINC) -fPIC -std=c++11 -I$(DDAS_INC) -I. -g \
	-I/usr/include/tcl8.6 -I$(SPECINC)
CXXLDFLAGS=-g -L$(DAQLIB) -L$(SPECLIB)	\
	-lSwTrigger -L$(DDAS_LIB) -lddasformat \
	-ltclPlus -lException -lTclGrammerApp -lDDASUnpacker \
	-ldaqio -ldataformat -lDataFlow -lurl -ldaqshm \
	-Wl,-rpath=$(DDAS_LIB) -Wl,-rpath=$(DAQLIB) \
	-Wl,-rpath=$(SPECLIB)	\
	-lgsl -lgslcblas -ltcl8.6 -Wl,-z,defs

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $^



all: libFitter.so libFitEditor.so

libFitter.so: FitExtender.o functions.o lmfit.o FitHitUnpacker.o \
	TCLDDASFitHitUnpacker.o StripTrace.o ThresholdClassifier.o
	$(CXX) -o libFitter.so -shared $^ $(CXXLDFLAGS)
	echo  pkg_mkIndex -verbose . libFitter.so | tclsh


libFitEditor.so: FitEditor.o functions.o lmfit.o FitHitUnpacker.o
	$(CXX) -o libFitEditor.so -shared $^ $(CXXLDFLAGS)

#
#  Requires PREFIX be defined and pointing to installtion top level dir e.g.:
#
# make install PREFIX=/usr/opt/ddastoys
#
install:
	install -d $(PREFIX)
	install -d $(PREFIX)/include
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/bin
	install -d $(PREFIX)/share
	install -m 0755 libFitter.so $(PREFIX)/lib
	install -m 0644 pkgIndex.tcl $(PREFIX)/lib
	install -m 0755 libFitEditor.so $(PREFIX)/lib
	install -m 0644 *.h $(PREFIX)/include
	install -m 0755 tracepeek.tcl $(PREFIX)/bin/tracepeek
	install -m 0644 AREADME.txt $(PREFIX)/share

clean:
	rm -f *.so *.o pkgIndex.tcl

