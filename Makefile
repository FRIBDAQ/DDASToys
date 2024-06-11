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
#    approprite thisroot.sh script in /usr/opt/root/root-x.yy-zz/bin.
#  - A version of Gnu Scientific Library (gsl) is installed, we expect to
#    find it in /usr/lib/x86_64-linux-gnu otherwise you may have to edit
#    the Makefile to point at your GSL headers/libraries.

# Set the top level install path if not provided:

DEFAULT_PREFIX=/usr/opt/ddastoys
ifeq ($(PREFIX),)
$(info No prefix specified, assuming $(DEFAULT_PREFIX))
PREFIX=$(DEFAULT_PREFIX)
endif

# Check for cmake of the appropriate version. Requires at least cmake 3.15
# which allows for overriding the default installation directory via --prefix
# flag at install time rather than requiring -DCMAKE_INSTALL_PREFIX when
# building. Ideally I'd like to port all of this over to a cmake build but
# thats a more significant amount of work.

# It's an error not to have cmake:
ifeq (, $(shell which cmake))
$(error No cmake in $(PATH), cmake 3.15+ is required to build DDASToys)
endif

# It's also an error if cmake version is 3.14 or earlier (e.g., on buster):
CMAKE_VERSION_GT_315=$(shell cmake --version | head -1 | cut -d " " -f 3 | awk -F. '$$1 >= 3 && $$2 > 15')
ifeq ($(CMAKE_VERSION_GT_315),)
$(error cmake version 3.15+ required but found $(shell cmake --version | head -1))
endif

# If no qmake, we can still build everything except traceview:
BUILD_TRACEVIEW=1
ifeq (, $(shell which qmake))
BUILD_TRACEVIEW=0
$(info Qt version 5.11+ is required to build traceview, skipping)
endif

QT_VERSION_GT_511=$(shell qmake -qt=5 --version | tail -1 | cut -d " " -f 4 | awk -F. '$$1 >= 5 && $$2 >= 11')
ifeq ($(QT_VERSION_GT_511),)
BUILD_TRACEVIEW=0
$(info Qt version 5.11+ required to build traceview but found $(shell qmake -qt=5 --version | tail -1 | cut -d " " -f 2-), skipping)
endif

# Now we actually get to the Making:

CXX = g++

MAXPOINTS = 200

# Format library install location (see .gitmodules):

FMTINC=$(PREFIX)/DDASFormat/include
FMTLIB=$(PREFIX)/DDASFormat/lib
FMTBUILDDIR=$(PWD)/DDASFormat/build

CXXFLAGS = -std=c++14 -g -O2 -Wall -I. -I$(DAQINC) -I$(FMTINC)
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
libs: libDDASFormat.so libFitEditorAnalytic.so libFitEditorTemplate.so \
	libDDASFitHitUnpacker.so
objs: CRingItemProcessor.o
subdirs: eeconverter
ifeq ($(BUILD_TRACEVIEW), 1)
subdirs: eeconverter traceview
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(EXTRACXXFLAGS) -c $^

traceview:
	(cd TraceView; /usr/bin/qmake -qt=5 traceview.pro FMTINC=$(FMTINC) FMTLIB=$(FMTLIB))
	$(MAKE) -C TraceView

eeconverter:
	FMTINC=$(FMTINC) FMTLIB=$(FMTLIB) $(MAKE) -C EEConverter

libDDASFormat.so:
	(mkdir -p $(FMTBUILDDIR); cd $(FMTBUILDDIR); cmake ..; $(MAKE); cmake --install . --prefix $(PREFIX)/DDASFormat)

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

	for f in $(shell find . -type f -name "*.so" ! -name "libDDASFormat.so"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.pcm"); do install -m 0755 $$f $(PREFIX)/lib ; done
	for f in $(shell find . -type f -name "*.rootmap"); do install -m 0755 $$f $(PREFIX)/lib ; done
	ln -sf $(PREFIX)/lib/DDASRootFit_rdict.pcm $(PREFIX)/bin/DDASRootFit_rdict.pcm

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
	rm -rf $(FMTBUILDDIR)
	$(MAKE) -C TraceView clean
	rm -f TraceView/traceview
	$(MAKE) -C EEConverter clean
	$(MAKE) -C Docs clean
