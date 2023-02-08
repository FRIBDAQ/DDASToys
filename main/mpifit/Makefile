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

MAXPOINTS = 200

SPECINC = $(SPECTCLHOME)/include
SPECLIB = $(SPECTCLHOME)/lib

CXXFLAGS = -std=c++11 -g -O2 -Wall -I. \
	-I$(DAQINC) -I$(DDAS_INC) -I/usr/include/tcl8.6 \
	-I$(SPECINC) -I$(XIAAPIDIR)/include
CXXLDFLAGS = -L$(DAQLIB) -L$(SPECLIB) \
	-lSwTrigger -L$(DDAS_LIB) -lddasformat \
	-ltclPlus -lException -lTclGrammerApp -lDDASUnpacker \
	-ldaqio -ldataformat -lDataFlow -lurl -ldaqshm \
	-lgsl -lgslcblas -ltcl8.6 

CUDACXXFLAGS = -DCUDA --compiler-options -fPIC \
	-I/usr/opt/libcudaoptimize/include -DMAXPOINTS=$(MAXPOINTS)
GNUCXXFLAGS = -fPIC

CUDALDFLAGS = -L/usr/opt/libcudaoptimize/lib -lCudaOptimize \
	--linker-options -rpath=$(DDAS_LIB) \
	--linker-options -rpath=$(DAQLIB) \
	--linker-options -rpath=$(SPECLIB) 

GNULDFLAGS = -Wl,-rpath=$(DDAS_LIB) -Wl,-rpath=$(DAQLIB) -Wl,-rpath=$(SPECLIB)

ifdef CUDA
CXX = nvcc
CUDAOBJ = CudaFitEngineAnalytic.o cudafit_analytic.o

.SUFFIXES: .cu
EXTRACXXFLAGS = $(CUDACXXFLAGS)
EXTRALDFLAGS = $(CUDALDFLAGS)
.cu.o:
	$(CXX) -c $(CXXFLAGS) $(EXTRACXXFLAGS) $^
else
EXTRACXXFLAGS = $(GNUCXXFLAGS)
EXTRALDFLAGS = $(GNULDFLAGS)
endif

LIBS = libFitEditorAnalytic.so libFitEditorTemplate.so libDDASFitHitUnpacker.so

all: $(LIBS) traceview

.cpp.o:
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

traceview:
	(cd TraceView; qmake -qt=5 traceview.pro; make)

libFitEditorAnalytic.so: FitEditorAnalytic.o Configuration.o \
	functions_analytic.o lmfit_analytic.o \
	CFitEngine.o SerialFitEngineAnalytic.o \
	DDASFitHitUnpacker.o \
	$(CUDAOBJ)
	$(CXX) -o libFitEditorAnalytic.so -shared $^ $(CXXLDFLAGS) $(EXTRALDFLAGS)

libFitEditorTemplate.so: FitEditorTemplate.o Configuration.o \
	functions_template.o lmfit_template.o \
	DDASFitHitUnpacker.o
	$(CXX) -o libFitEditorTemplate.so -shared $^ $(CXXLDFLAGS) $(EXTRALDFLAGS)

libDDASFitHitUnpacker.so: DDASFitHitUnpacker.o
	$(CXX) -o libDDASFitHitUnpacker.so -shared -z defs $^ $(CXXLDFLAGS) $(EXTRALDFLAGS)

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
	install -m 0644 *.h $(PREFIX)/include
	install -m 0755 TraceView/traceview $(PREFIX)/bin/traceview
	install -m 0644 AREADME.txt $(PREFIX)/share

clean:
	rm -f *.so *.o
	make -C TraceView clean
	rm -f TraceView/traceview
