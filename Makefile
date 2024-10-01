##
# A refactored NSCLDAQ version containing the DDAS software must be defined
# externally. This NSCLDAQ version must have the fragement index stuff built
# in, major version 12.x and higher. MAXPOINTS should be modified to match the
# longest trace we're fitting if this is CUDA enabled.
#
# Required definitions:
#  - cmake 3.15+ for building and installing the DDASFormat library and
#    header files.
#  - Qt 5.11+ for TraceView.
#  - A version of the refactored NSCLDAQ containing the old DDAS headers
#    and libraries (12.1+ for this branch).
#  - The same ROOT version used to compile the NSCLDAQ version used. These can
#    be found under /usr/opt/root, (probably) not a module file. Source the
#    approprite thisroot.sh script in /usr/opt/root/root-M.mm-ee/bin.
#  - A version of Gnu Scientific Library (gsl) is installed, we expect to
#    find it in /usr/lib/x86_64-linux-gnu otherwise you may have to edit
#    the Makefile to point at your GSL headers/libraries.
#  - UMFT: Base of unified format library installation. May or may not come
#    from the NSCLDAQ version DDASToys is compiled against.
#  - LibTorch: for the ML inference editors. The Makefile variables TORCHINC
#    and TORCHLIB point to the API headers/libraries. If the expected
#    directories do not exist, the build will skip building the ML stuff.
#
# To build: UFMT=/path/to/ufmt PREFIX=/path/to/install/dir make all install
#

# Set the top level install path if not provided:

DEFAULT_PREFIX=$(HOME)/ddastoys
ifeq ($(PREFIX),)
$(info No prefix specified, assuming $(DEFAULT_PREFIX))
PREFIX=$(DEFAULT_PREFIX)
endif

# cmake is required to build the DDASFormat library. Note the format library
# itself will enforce the cmake version requirement:

ifeq (, $(shell which cmake))
$(error No cmake found in $(PATH), cmake is required to build DDASToys)
endif

# traceview requires qmake, skip the build if its not installed:

BUILD_TRACEVIEW=1
ifeq (, $(shell which qmake))
BUILD_TRACEVIEW=0
$(info Qt version 5.11+ is required to build traceview, skipping...)
endif

QT_VERSION_GT_511=$(shell qmake -qt=5 --version | tail -1 | cut -d " " -f 4 | awk -F. '$$1 >= 5 && $$2 >= 11')
ifeq ($(QT_VERSION_GT_511),)
BUILD_TRACEVIEW=0
$(info Qt version 5.11+ required to build traceview but found $(shell qmake -qt=5 --version | tail -1 | cut -d " " -f 2-), skipping...)
endif


# We have to check for Torch stuff here, and skip building the inference model
# fit editor if LibTorch isn't installed where we think it should be. Edit for
# your install path if needed:

TORCHINC=/usr/include/torch/csrc/api/include 
TORCHLIB=/usr/include/torch/csrc/api/lib 

BUILD_MLINFERENCE=1
ifeq ($(shell test -d $(TORCHINC) && echo 1 || echo 0), 0)
BUILD_MLINFERENCE=0
$(info LibTorch is required to build the ML inference editor, skipping...)
endif

# Now we actually get to the Making:

CXX=g++

MAXPOINTS=250

# Unified format library:

UFMTINC=$(UFMT)/include
UFMTLIB=$(UFMT)/lib

# DDAS format library (see .gitmodules):

DDASFMTPATH=$(PREFIX)/DDASFormat
DDASFMTINC=$(DDASFMTPATH)/include
DDASFMTLIB=$(DDASFMTPATH)/lib
DDASFMTBUILDDIR=$(PWD)/DDASFormat/build

# Flags depend on whether we build for GPU fitting:

CXXFLAGS=-std=c++14 -g -O2 -I. -I$(DDASFMTINC) -I$(UFMTINC)
CXXLDFLAGS=-lgsl -lgslcblas -L$(DDASFMTLIB) -lDDASFormat

CUDACXXFLAGS=-DCUDA --compiler-options -fPIC 				\
	-I/usr/opt/libcudaoptimize/include -DMAXPOINTS=$(MAXPOINTS)
CUDALDFLAGS=-L/usr/opt/libcudaoptimize/lib -lCudaOptimize 		\
	--linker-options -rpath=$(DDASFMTLIB)

GNUCXXFLAGS=-fPIC -Wall
GNULDFLAGS=-Wl,-rpath=$(DDASFMTLIB)

ifdef CUDA
CXX = nvcc
CUDAOBJ = CudaFitEngineAnalytic.o cudafit_analytic.o

.SUFFIXES: .cu
EXTRACXXFLAGS=$(CUDACXXFLAGS)
EXTRALDFLAGS=$(CUDALDFLAGS)
.cu.o:
	$(CXX) -c $(CXXFLAGS) $(EXTRACXXFLAGS) $^
else
EXTRACXXFLAGS=$(GNUCXXFLAGS)
EXTRALDFLAGS=$(GNULDFLAGS)
endif

##
# Build order matters: eeconverter and traceview require that
# DDASFitHitUnpacker.o and CRingItemProcessor.o exist.
#

all: exec docs
exec: libs objs subdirs
libs: libDDASFormat.so libDDASFitHitUnpacker.so libFitEditorAnalytic.so \
	libFitEditorTemplate.so libDDASFitHitUnpacker.so
ifeq ($(BUILD_MLINFERENCE), 1)
libs: libDDASFormat.so libDDASFitHitUnpacker.so libFitEditorAnalytic.so \
	libFitEditorTemplate.so libFitEditorMLInference.so 		\
	libDDASFitHitUnpacker.so
endif
objs: CRingItemProcessor.o
subdirs: eeconverter
ifeq ($(BUILD_TRACEVIEW), 1)
subdirs: eeconverter traceview
endif

libDDASFormat.so:
	(mkdir -p $(DDASFMTBUILDDIR); cd $(DDASFMTBUILDDIR); cmake .. -DCMAKE_INSTALL_PREFIX=$(DDASFMTPATH); $(MAKE); $(MAKE) install)

libFitEditorAnalytic.so: FitEditorAnalytic.o Configuration.o 		\
	functions_analytic.o lmfit_analytic.o 				\
	CFitEngine.o SerialFitEngineAnalytic.o 				\
	$(CUDAOBJ)
	$(CXX) -o libFitEditorAnalytic.so -shared $^ 			\
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

libFitEditorTemplate.so: FitEditorTemplate.o Configuration.o 		\
	functions_template.o lmfit_template.o 
	$(CXX) -o libFitEditorTemplate.so -shared $^ 			\
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

libFitEditorMLInference.so: FitEditorMLInference.o Configuration.o 	\
	functions_analytic.o mlinference.o
	$(CXX) -o libFitEditorMLInference.so -shared $^			\
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

libDDASFitHitUnpacker.so: DDASFitHitUnpacker.o
	$(CXX) -o libDDASFitHitUnpacker.so -shared -z defs $^ 		\
	$(CXXLDFLAGS) $(EXTRALDFLAGS)

FitEditor%.o: FitEditor%.cpp
	$(CXX) -I$(DAQINC) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

FitEditorMLInference.o: FitEditorMLInference.cpp
	$(CXX) -I$(DAQINC) -I$(TORCHINC) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^ \
	-L$(TORCHLIB) -Wl,-rpath=$(TORCHLIB) -ltorch -ltorch_cpu -lc10

mlinference.o: mlinference.cpp
	$(CXX) $(CXXFLAGS) -I$(TORCHINC) $(EXTRACXXFLAGS) -c $^ 	\
	-L$(TORCHLIB) -Wl,-rpath=$(TORCHLIB) -ltorch -ltorch_cpu -lc10

CRingItemProcessor.o: CRingItemProcessor.cpp
	$(CXX) -I$(DAQINC) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

traceview:
	(cd TraceView; /usr/bin/qmake -qt=5 traceview.pro DDASFMTINC=$(DDASFMTINC) DDASFMTLIB=$(DDASFMTLIB))
	$(MAKE) -C TraceView

eeconverter:
	DDASFMTINC=$(DDASFMTINC) DDASFMTLIB=$(DDASFMTLIB) $(MAKE) -C EEConverter

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

	for f in $(shell find . -type f -name "*.so" ! -name "libDDASFormat.so"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.pcm"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.rootmap"); do install -m 0755 $$f $(PREFIX)/lib ; done
	ln -sf $(PREFIX)/lib/DDASRootFitFormat_rdict.pcm $(PREFIX)/bin/DDASRootFitFormat_rdict.pcm

	for f in $(shell find . -maxdepth 1 -type f -name "*.h" ! -name "CRingItemProcessor.h"); do install -m 0644 $$f $(PREFIX)/include; done
	for f in $(shell find ./EEConverter -type f -name "*Root*.h" ! -name "ProcessToRootSink.h"); do install -m 0644 $$f $(PREFIX)/include; done

	install -m 0755 EEConverter/eeconverter $(PREFIX)/bin/eeconverter
ifeq ($(BUILD_TRACEVIEW), 1)
	install -m 0755 TraceView/traceview $(PREFIX)/bin/traceview
endif

	cp -r Docs/manual $(PREFIX)/share/
	cp -r Docs/sourcedocs $(PREFIX)/share/

clean:
	rm -f *.so *.o
	rm -rf $(DDASFMTBUILDDIR)
	$(MAKE) -C TraceView clean
	rm -f TraceView/traceview
	$(MAKE) -C EEConverter clean
	$(MAKE) -C Docs clean
