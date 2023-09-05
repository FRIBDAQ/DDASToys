# A refactored NSCLDAQ version containing the DDAS software must be defined
# externally. This NSCLDAQ version must have the fragement index stuff built
# in, major version 12.x and higher. MAXPOINTS should be modified to match the
# longest trace we're fitting if this is CUDA enabled.
#
# Required definitions:
#  - A version of the refactored NSCLDAQ containing the old DDAS headers
#    and libraries has been set up.
#  - The same ROOT version used to compile the NSCLDAQ version used. These can
#    be found under /usr/opt/root, (probably) not a module file. Source the
#    approprite thisroot.sh script in /usr/opt/root/root-x.yy-zz/bin.
#  - A version of Gnu Scientific Library (gsl) is installed, we expect to
#    find it in /usr/lib/x86_64-linux-gnu otherwise you may have to edit
#    the Makefile to point at your GSL headers/libraries.

CXX = g++

MAXPOINTS = 200

CXXFLAGS = -std=c++11 -g -O2 -Wall -I. -I$(DAQINC) 
CXXLDFLAGS = -lgsl -lgslcblas -L$(DAQLIB) -lddasformat

CUDACXXFLAGS = -DCUDA --compiler-options -fPIC \
	-I/usr/opt/libcudaoptimize/include -DMAXPOINTS=$(MAXPOINTS)
CUDALDFLAGS = -L/usr/opt/libcudaoptimize/lib -lCudaOptimize \
	--linker-options -rpath=$(DAQLIB)

GNUCXXFLAGS = -fPIC
GNULDFLAGS = -Wl,-rpath=$(DAQLIB) 

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

all: exec docs
exec: libs traceview
libs: libFitEditorAnalytic.so libFitEditorTemplate.so libDDASFitHitUnpacker.so

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

traceview:
	(cd TraceView; /usr/bin/qmake -qt=5 traceview.pro)
	$(MAKE) -C TraceView

libFitEditorAnalytic.so: FitEditorAnalytic.o Configuration.o \
	functions_analytic.o lmfit_analytic.o \
	CFitEngine.o SerialFitEngineAnalytic.o \
	$(CUDAOBJ)
	$(CXX) -o libFitEditorAnalytic.so -shared $^ \
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

libFitEditorTemplate.so: FitEditorTemplate.o Configuration.o \
	functions_template.o lmfit_template.o 
	$(CXX) -o libFitEditorTemplate.so -shared $^ \
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

libDDASFitHitUnpacker.so: DDASFitHitUnpacker.o
	$(CXX) -o libDDASFitHitUnpacker.so -shared -z defs $^ \
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

#
# Build docbooks and doxygen documentation
#

docs:
	$(MAKE) -C Docs

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
	cp -r Docs/manual $(PREFIX)/share/
	cp -r Docs/sourcedocs $(PREFIX)/share/

clean:
	rm -f *.so *.o
	$(MAKE) -C TraceView clean
	rm -f TraceView/TraceView
	$(MAKE) -C Docs clean
