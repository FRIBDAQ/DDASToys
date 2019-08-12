# SPECTCLHOME must be define externally to point to a SpecTcl that has
# the fragment index stuff built in.

SPECINC=$(SPECTCLHOME)/include
SPECLIB=$(SPECTCLHOME)/lib

CXXFLAGS=-I$(DAQINC) -fPIC -std=c++11 -I$(DDAS_INC) -I. -g \
	-I/usr/include/tcl8.6 -I$(SPECINC)
CXXLDFLAGS=-g -L$(DAQLIB) -L$(SPECLIB)	\
	-lSwTrigger -L$(DDAS_LIB) -lddasformat \
	-ltclPlus -lException -lTclGrammerApp  \
	-ldaqio -ldataformat -lDataFlow -lurl -ldaqshm \
	-Wl,-rpath=$(DDAS_LIB) -Wl,-rpath=$(DAQLIB) \
	-Wl,-rpath=$(SPECLIB)	\
	-lgsl -lgslcblas -ltcl8.6 -Wl,-z,defs

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $^



all: libFitter.so

libFitter.so: FitExtender.o functions.o lmfit.o FitHitUnpacker.o \
	TCLDDASFitHitUnpacker.o
	$(CXX) -o libFitter.so -shared $^ $(CXXLDFLAGS)
	echo  pkg_mkIndex -verbose . *.so | tclsh

clean:
	rm -f *.so *.o pkgIndex.tcl

