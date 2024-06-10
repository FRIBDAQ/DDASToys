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

FMTINC=./DDASFormat
FMTLIB=./DDASFormat

CXXFLAGS = -std=c++11 -g -O2 -Wall -I. -I$(DAQINC) -I$(FMTINC)
CXXLDFLAGS = -lgsl -lgslcblas -L$(FMTLIB) -lDDASFormat

CUDACXXFLAGS = -DCUDA --compiler-options -fPIC \
	-I/usr/opt/libcudaoptimize/include -DMAXPOINTS=$(MAXPOINTS)
CUDALDFLAGS = -L/usr/opt/libcudaoptimize/lib -lCudaOptimize \
	--linker-options -rpath=$(FMTLIB)

GNUCXXFLAGS = -fPIC
GNULDFLAGS = -Wl,-rpath=$(FMTLIB)

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

##
# Build order matters: eeconverter and traceview require that
# DDASFitHitUnpacker.o and CRingItemProcessor.o exist.
#

all: exec docs
exec: libs objs subdirs
libs: libDDASFormat.so libFitEditorAnalytic.so libFitEditorTemplate.so libDDASFitHitUnpacker.so
objs: CRingItemProcessor.o
subdirs: eeconverter traceview

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

traceview:
	(cd TraceView; /usr/bin/qmake -qt=5 traceview.pro)
	$(MAKE) -C TraceView

eeconverter:
	$(MAKE) -C EEConverter

libDDASFormat.so:
	(cd DDASFormat; mkdir -p build; cd build; cmake --build ..; $(MAKE))

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

##
# Build docbooks and doxygen documentation
#

docs:
	$(MAKE) -C Docs

##
# Requires PREFIX be defined and pointing to installtion top level dir e.g.:
#
# make install PREFIX=/usr/opt/ddastoys
#

install:
	install -d $(PREFIX)
	install -d $(PREFIX)/include
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/bin
	install -d $(PREFIX)/share

	for f in $(shell find . -type f -name "*.so"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.pcm"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.rootmap"); do install -m 0755 $$f $(PREFIX)/lib ; done
	ln -sf $(PREFIX)/lib/DDASRootFit_rdict.pcm $(PREFIX)/bin/DDASRootFit_rdict.pcm

	for f in $(shell find . -maxdepth 1 -type f -name "*.h" ! -name "CRingItemProcessor.h"); do install -m 0644 $$f $(PREFIX)/include; done
	for f in $(shell find ./EEConverter -type f -name "*Root*.h" ! -name "ProcessToRootSink.h"); do install -m 0644 $$f $(PREFIX)/include; done

	install -m 0755 EEConverter/eeconverter $(PREFIX)/bin/eeconverter
	install -m 0755 TraceView/traceview $(PREFIX)/bin/traceview

	cp -r Docs/manual $(PREFIX)/share/
	cp -r Docs/sourcedocs $(PREFIX)/share/

clean:
	rm -f *.so *.o
	(cd DDASFormat; cmake --build . --target clean)
	$(MAKE) -C TraceView clean
	rm -f TraceView/traceview
	$(MAKE) -C EEConverter clean
	$(MAKE) -C Docs clean
