# Compiler and flags:

CC = gcc
CXX = g++

UFMTINC=$(UFMT)/include
UFMTLIB=$(UFMT)/lib

# Assume DAQ environment is set up, DDASFMT stuff from toplevel Makefile:

CXXFLAGS = -Wall -g -O2 -fPIC -I.. -I$(DDASFMTINC) -I$(DAQINC) -I$(UFMTINC)
LDFLAGS = -L$(DDASFMTLIB) -Wl,-rpath=$(DDASFMTLIB) 		\
	-L$(DAQLIB) -Wl,-rpath=$(DAQLIB) 			\
	-L$(UFMTLIB) -Wl,-rpath=$(UFMTLIB)
LDLIBS = -lDDASFormat -ldataformat -lException -lurl -ldaqio -lAbstractFormat 

# Flags related to CERN ROOT:

ROOTCFLAGS  = $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLDFLAGS = $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

all: eeconverter libDDASRootFit.so

eeconverter: eeconverter.o converterargs.o RootFileDataSink.o 		\
	ProcessToRootSink.o CEventProcessor.o StringsToIntegers.o 	\
	DDASRootFitHit.o DDASRootFitEvent.o DDASRootFit.o 		\
	../DDASFitHitUnpacker.o ../CRingItemProcessor.o
	$(CXX) $(CXXFLAGS) $(ROOTCFLAGS) -o $@ $^ 			\
	$(LDFLAGS) $(LDLIBS) $(ROOTLDFLAGS)

libDDASRootFit.so: DDASRootFit.o DDASRootFitHit.o DDASRootFitEvent.o
	$(CXX) -fPIC -shared -o $@ $(CXXFLAGS) $^ 			\
	-L$(DDASFMTLIB) -Wl,-rpath=$(DDASFMTLIB) -lDDASFormat

DDASRootFit.cpp: DDASRootFitEvent.h DDASRootFitHit.h ../DDASFitHit.h 	\
	RootExtensions.h LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ 	-rml libDDASRootFit.so 		\
	-rmf libDDASRootFit.rootmap -I.. -I$(DDASFMTINC) $^

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

converterargs.o:
	gengetopt < converterargs.ggo --file-name=converterargs
	$(CC) -c converterargs.c

.PHONY: clean cleanest

clean:
	rm -f *.o *.so *.pcm *.rootmap
	rm -f DDASRootFit.cpp
	rm -f converterargs.c converterargs.h
	rm -f eeconverter
