CC      = g++
CFLAGS  = -std=c++11 -g  -O2
LDFLAGS = -pthread -Wall -lrt  -lzmq  -g

#  Assume DAQ environment is set up:
#  Use DDAS-2.2 damnit:

DDASROOT=/usr/opt/ddas/2.2
DDASINC=$(DDASROOT)/include
DDASLIB=$(DDASROOT)/lib

CXXFLAGS=$(CFLAGS) -I$(DAQROOT)/include -I$(DDAS_INC) -fPIC
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

all:   ringblockdealer libDDASFitClasses.so  tests 

tests: functiontests pulsemaker fit1 fit2 TestAnalyzer v10to11converter \
	DisplayExtensions

CRingFileReader.o: CRingFileReader.cpp
	$(CC) -c $(CXXFLAGS) $<

CRingFileBlockReader.o: CRingFileBlockReader.cpp
	$(CC) -c $(CXXFLAGS) $<



ringblockdealer: ringblockdealer.o CRingFileBlockReader.o CDDASAnalyzer.o \
	CRingFileOutputter.o FitHitUnpacker.o 				\
	functions.o zmqsortedwriter.o CZMQRingOutputter.o CSortingOutputter.o \
	CRootFileDataSink.o DDASRootFitEvent.o DDASRootFitHit.o \
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

CRootFileDataSink.o: CRootFileDataSink.cpp CRootFileDataSink.cpp
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

RootFitDictionary.h: DDASRootFitEvent.h DDASRootFitHit.h RootExtensions.h \
	LinkDef.h
	LD_LIBRARY_PATH=$(ROOTSYS)/lib $(ROOTSYS)/bin/rootcint 	\
		-f RootFitDictionary.cpp  \
		-rml libDDASFitClasses.so \
		-rmf libDDASFitClasses.rootmap  \
		-I$(DDASINC)	\
		$^


functions.o: functions.cpp functions.h
	$(CXX) -c $(CXXFLAGS) $<


functiontests: functiontests.o functions.o
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

functiontests.o: functiontests.cpp functions.h


pulsemaker: pulsemaker.o functions.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

fit1: fit1.cpp functions.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas

fit2: fit2.cpp functions.o
	$(CXX) $(CXXFLAGS) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas


ringthreadblockdealer: ringthreadblockdealer.o CRingThreadedBlockReader.o


TestAnalyzer: TestAnalyzer.o CDDASAnalyzer.o functions.o FitHitUnpacker.o 
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas -ldataformat -ldaqio -lDataFlow -lurl -lException

v10to11converter: v10to11converter.o
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -ldataformat -ldaqio -lDataFlow -lurl -lException

DisplayExtensions : DisplayExtensions.o FitHitUnpacker.o functions.o 
	$(CXX) $(CXXLDFLAGS) -o $@ $^ -lgsl -lgslcblas -ldataformat -ldaqio -lDataFlow -lurl -lException




libDDASFitClasses.so:  DDASRootFitHit.o DDASRootFitEvent.o  \
	RootFitDictionary.o RootExtensions.o RootFitDictionary.o
	$(CXX) -shared -o $@ $^  \
	-L$(DDASLIB)  -lddasformat \
	-Wl,-rpath=$(DDASLIB)


.PHONY: clean cleanest

clean:
	rm -f *.o  \
	functiontests pulsemaker fit1 fit2 TestAnalyzer v10to11converter \
	DisplayExtensions \
	ringblockdealer libDDASFitClasses.so
