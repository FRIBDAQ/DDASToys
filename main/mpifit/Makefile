# SPECTCLHOME must be define externally to point to a SpecTcl that has
# the fragment index stuff built in. Furthermore, an NSCLDAQ 11.4 or higher
# must have been set up (daqsetup.bash sourced in).
# XIAAPIDIR must be defined to point to the XIA API directory used by that
# NSCLDAQ distro.



#  Note MAXPOINTS should  be modified to match the longest trace we're fitting
#  if this is CUDA enabled.

#  Required definitions:
#  SPECTCLHOME - Base of SpecTcl.
#  If SpecTcl 5.10 or later is used, the root for that version must also be setupe
#  (e.g. /usr/opt/root/some-version/bin/thisroot.sh).
#  A version of NSCLDAQ has been set up.
#  A version of DDAS has been set up.
#  Define XIAAPI to point to the XIAAPI version used by your NSCLDAQ setup.

MAXPOINTS=200

SPECINC=$(SPECTCLHOME)/include
SPECLIB=$(SPECTCLHOME)/lib

CXXFLAGS=-I$(DAQINC)  -std=c++11 -I$(DDAS_INC) -I. -g \
	-I/usr/include/tcl8.6 -I$(SPECINC) -I$(XIAAPIDIR)/include
CXXLDFLAGS=-g -L$(DAQLIB) -L$(SPECLIB)	\
	-lSwTrigger -L$(DDAS_LIB) -lddasformat \
	-ltclPlus -lException -lTclGrammerApp -lDDASUnpacker \
	-ldaqio -ldataformat -lDataFlow -lurl -ldaqshm \
	-lgsl -lgslcblas -ltcl8.6 

CUDACXXFLAGS= -DCUDA --compiler-options -fPIC -I/usr/opt/libcudaoptimize/include \
	-DMAXPOINTS=$(MAXPOINTS)
GNUCXXFLAGS=  -fPIC

CUDALDFLAGS= 	-L/usr/opt/libcudaoptimize/lib -lCudaOptimize \
	--linker-options -rpath=$(DDAS_LIB) \
	--linker-options -rpath=$(DAQLIB)        \
	--linker-options -rpath=$(SPECLIB) 

GNULDFLAGS=-Wl,-rpath=$(DDAS_LIB) -Wl,-rpath=$(DAQLIB) \
	-Wl,-rpath=$(SPECLIB)

ifdef CUDA
CXX=nvcc
CUDAOBJ=CudaFitEngine.o cudafit.o

.SUFFIXES: .cu
EXTRACXXFLAGS=$(CUDACXXFLAGS)
EXTRALDFLAGS=$(CUDALDFLAGS)
.cu.o:
	$(CXX) -c $(CXXFLAGS) $(EXTRACXXFLAGS) $^
else
EXTRACXXFLAGS=$(GNUCXXFLAGS)
EXTRALDFLAGS=$(GNULDFLAGS)
#CXXLDFLAGS+=-#	-Wl,-z,defs
endif


.cpp.o:
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

all: libFitter.so libFitEditor.so libTclUnpacker.so


TCLDDASFitHitUnpacker.o: TCLDDASFitHitUnpacker.cpp
	g++ -c $^ $(CXXFLAGS) $(GNUCXXFLAGS)


libFitter.so: FitExtender.o functions.o lmfit.o FitEngine.o serialJacobian.o \
	FitHitUnpacker.o \
	TCLDDASFitHitUnpacker.o StripTrace.o ThresholdClassifier.o \
	$(CUDAOBJ)
	$(CXX) -o libFitter.so -shared $^ $(CXXLDFLAGS) $(EXTRALDFLAGS)



libTclUnpacker.so: TCLDDASFitHitUnpacker.o
	g++ -o libTclUnpacker.so -shared $^ $(CXXLDFLAGS) $(GNULDFLAGS)
	echo  pkg_mkIndex -verbose . libTclUnpacker.so | tclsh

libFitEditor.so: FitEditor.o functions.o lmfit.o FitEngine.o serialJacobian.o \
	FitHitUnpacker.o $(CUDAOBJ)
	$(CXX) -o libFitEditor.so -shared $^ $(CXXLDFLAGS) $(EXTRALDFLAGS)

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
	install -m 0755 *.so $(PREFIX)/lib
	install -m 0644 pkgIndex.tcl $(PREFIX)/lib
	install -m 0644 *.h $(PREFIX)/include
	install -m 0755 tracepeek.tcl $(PREFIX)/bin/tracepeek
	install -m 0644 AREADME.txt $(PREFIX)/share

clean:
	rm -f *.so *.o pkgIndex.tcl

