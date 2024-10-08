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

all: eeconverter libDDASRootFitFormat.so

eeconverter: eeconverter.o converterargs.o RootFileDataSink.o 		\
	ProcessToRootSink.o CEventProcessor.o StringsToIntegers.o 	\
	DDASRootFitHit.o DDASRootFitEvent.o DDASRootFitFormat.o 	\
	../DDASFitHitUnpacker.o ../CRingItemProcessor.o
	$(CXX) $(CXXFLAGS) $(ROOTCFLAGS) -o $@ $^ 			\
	$(LDFLAGS) $(LDLIBS) $(ROOTLDFLAGS)

libDDASRootFitFormat.so: DDASRootFitFormat.o DDASRootFitHit.o 		\
	DDASRootFitEvent.o
	$(CXX) -fPIC -shared -o $@ $(CXXFLAGS) $^ 			\
	-L$(DDASFMTLIB) -Wl,-rpath=$(DDASFMTLIB) -lDDASFormat

DDASRootFitFormat.cpp: DDASRootFitEvent.h DDASRootFitHit.h 		\
	../DDASFitHit.h RootExtensions.h LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ 	-rml libDDASRootFitFormat.so 	\
	-rmf libDDASRootFitFormat.rootmap -I.. -I$(DDASFMTINC) $^

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

converterargs.o:
	gengetopt < converterargs.ggo --file-name=converterargs
	$(CC) -c converterargs.c

.PHONY: clean cleanest

clean:
	rm -f *.o *.so *.pcm *.rootmap
	rm -f DDASRootFitFormat.cpp
	rm -f converterargs.c converterargs.h
	rm -f eeconverter
