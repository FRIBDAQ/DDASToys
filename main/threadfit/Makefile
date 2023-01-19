COMMON_TIMING=0

# Assume some GSL is set up:

GSLCFLAGS=$(shell gsl-config --cflags)
GSLLDFLAGS=$(shell gsl-config --libs)
GSLLIB=$(shell gsl-config --prefix)/lib

C      = g++
CFLAGS  = -std=c++11 -g  -O2 $(GSLCFLAGS)
LDFLAGS = $(GSLLDFLAGS) -pthread -Wall -lrt  -lzmq  -g -Wl,-rpath=$(GSLLIB)

# Assume DAQ environment is set up:

DDASROOT=$(DDAS_ROOT)
DDASINC=$(DDASROOT)/include
DDASLIB=$(DDASROOT)/lib

CXXFLAGS=$(CFLAGS) -I$(DAQROOT)/include -I$(DDAS_INC) -fPIC \
	-DCOMMON_TIMING=$(COMMON_TIMING)
CXXLDFLAGS=$(LDFLAGS) -L$(DAQLIB)  -L$(DDASLIB) \
	-lFragmentIndex			\
	-ldataformat			\
	-lddasformat			\
	-lException			\
	-Wl,"-rpath=$(DAQLIB)" -Wl,"-rpath=$(DDASLIB)"

# Flags related to CERN ROOT

ROOTCFLAGS=$(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLDFLAGS=$(shell $(ROOTSYS)/bin/root-config --libs --ldflags)
ROOT_RPATH=-Wl,-rpath=$(ROOTSYS)/lib/root

#

all:   ringblockdealer libDDASFitClasses.so asciiconvert binconvert  tests 


pulsemaker:
	(cd pulsemacher; make)
	mv pulsemacher/pulsemaker pulsemaker

tests: functiontests pulsemaker fit1 fit2 fit2t  TestAnalyzer v10to11converter \
	DisplayExtensions

CRingFileReader.o: CRingFileReader.cpp
	$(CC) -c $(CXXFLAGS) $<

CRingFileBlockReader.o: CRingFileBlockReader.cpp
	$(CC) -c $(CXXFLAGS) $<


asciiconvert: asciiconvert.cpp
	$(CXX) $(CXXFLAGS) -o asciiconvert asciiconvert.cpp $(CXXLDFLAGS) \
	-ldaqio

binconvert: binconvert.cpp
	$(CXX) $(CXXFLAGS) -DSTDINSOURCE_SUPPORTED -o binconvert binconvert.cpp $(CXXLDFLAGS) \
	-ldaqio -lurl


ringblockdealer: ringblockdealer.o CRingFileBlockReader.o CDDASAnalyzer.o \
	CRingFileOutputter.o FitHitUnpacker.o 				\
	functions.o lmfit.o \
	zmqsortedwriter.o CZMQRingOutputter.o CSortingOutputter.o \
	CRootFileDataSink.o CRootSelectableDataSink.o \
	COutputFormatFactory.o \
	DDASRootFitEvent.o DDASRootFitHit.o \
	RootFitDictionary.o RootExtensions.o RootFitDictionary.o
	$(CXX) $(CXXLDFLAGS) -o $@ $^ \
	-ldataformat -ldaqio -lDataFlow -lurl -lgsl -lgslcblas \
	$(ROOTLDFLAGS) $(ROOT_RPATH)

ringblockdealer.o: ringblockdealer.cpp
	$(CXX) -c $(CXXFLAGS) $<

CDDASAnalyzer.o: CDDASAnalyzer.cpp  CDDASAnalyzer.h Outputter.h
	$(CXX) -c $(CXXFLAGS) $<

DDASFileOutputter.o: DDASFileOutputter.cpp DDASFileOutputter.h Outputter.h \
	CDDASAnalyzer.h
	$(CXX) -c $(CXXFLAGS) $<

CRootFileDataSink.o: CRootFileDataSink.cpp CRootFileDataSink.h
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

CRootSelectableDataSink.o: CRootSelectableDataSink.cpp CRootSelectableDataSink.h
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

COutputFormatFactory.o: COutputFormatFactory.cpp COutputFormatFactory.h
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

DDASRootFitEvent.o: DDASRootFitEvent.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

DDASRootFitHit.o: DDASRootFitHit.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

RootExtensions.o: RootExtensions.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

RootFitDictionary.o: RootFitDictionary.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

zmqsortedwriter.o: zmqsortedwriter.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

RootFitDictionary.cpp: RootFitDictionary.h

RootFitDictionary.h: DDASRootFitEvent.h DDASRootFitHit.h RootExtensions.h rootTreeClasses.h functions.h  \
	LinkDef.h
	LD_LIBRARY_PATH=$(ROOTSYS)/lib $(ROOTSYS)/bin/rootcint 	\
		-f RootFitDictionary.cpp  \
		-rml libDDASFitClasses.so \
		-rmf libDDASFitClasses.rootmap  \
		-I$(DDASINC)	\
		$^


functions.o: functions.cpp functions.h
	$(CXX) -c $(CXXFLAGS) $<

lmfit.o: lmfit.cpp lmfit.h
	$(CXX) -c $(CXXFLAGS) $<

functiontests: functiontests.o functions.o lmfit.o
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

functiontests.o: functiontests.cpp functions.h




fit1: fit1.cpp functions.o lmfit.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

fit2: fit2.cpp functions.o lmfit.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

fit2t: fit2t.cpp functions.o lmfit.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas


ringthreadblockdealer: ringthreadblockdealer.o CRingThreadedBlockReader.o


TestAnalyzer: TestAnalyzer.o CDDASAnalyzer.o functions.o lmfit.o \
	FitHitUnpacker.o 
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas -ldataformat -ldaqio -lDataFlow -lurl -lException

v10to11converter: v10to11converter.o
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -ldataformat -ldaqio -lDataFlow -lurl -lException

DisplayExtensions : DisplayExtensions.o FitHitUnpacker.o functions.o 
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -ldataformat -ldaqio -lDataFlow -lurl -lException


libDDASFitClasses.so:  DDASRootFitHit.o DDASRootFitEvent.o  \
	RootFitDictionary.o RootExtensions.o RootFitDictionary.o functions.o
	$(CXX) -shared -o $@ $^  \
	-L$(DDASLIB)  -lddasformat \
	-Wl,-rpath=$(DDASLIB)


.PHONY: clean cleanest

clean:
	rm -f *.o  \
	functiontests  fit1 fit2 TestAnalyzer v10to11converter \
	DisplayExtensions \
	rm -f ringblockdealer libDDASFitClasses.so RootFitDictionary.* binconvert
	make -C pulsemacher clean
	rm -r pulsemaker
