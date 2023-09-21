# Compiler and flags:

CC = gcc
CXX = g++

# Assume DAQ environment is set up -- need DDAS verson with lddasfitformat!

CXXFLAGS = -Wall -g -O2 -fPIC -I.. -I$(SPECTCLINC) -I$(DAQINC)
LDFLAGS = -L$(SPECTCLLIB) -Wl,-rpath=$(SPECTCLLIB) -L$(DAQLIB) -Wl,-rpath=$(DAQLIB)
LDLIBS = -lTclGrammerApp -ldataformat -lddasformat -lException -lurl -ldaqio

# Flags related to CERN ROOT:

ROOTCFLAGS  = $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLDFLAGS = $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

all: eeconverter libDDASRootFit.so

eeconverter: eeconverter.o converterargs.o RootFileDataSink.o \
	ProcessToRootSink.o CEventProcessor.o StringsToIntegers.o \
	DDASRootFitHit.o DDASRootFitEvent.o DDASRootFit.o \
	RootExtensions.o ../DDASFitHitUnpacker.o ../CRingItemProcessor.o
	$(CXX) $(CXXFLAGS) $(ROOTCFLAGS) -o $@ $^ \
	$(LDFLAGS) $(LDLIBS) $(ROOTLDFLAGS)

libDDASRootFit.so: DDASRootFit.o DDASRootFitHit.o DDASRootFitEvent.o \
	RootExtensions.o
	$(CXX) -fPIC -shared -o $@ $(CXXFLAGS) $^ $(LDFLAGS) -lddasformat

DDASRootFit.cpp: DDASRootFitEvent.h DDASRootFitHit.h ../DDASFitHit.h \
	RootExtensions.h LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ \
	-rml libDDASRootFit.so -rmf libDDASRootFit.rootmap \
	-I$(DAQINC) $^

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $(ROOTCFLAGS) $<

converterargs.o:
	gengetopt < converterargs.ggo --file-name=converterargs
	$(CC) -c converterargs.c

.PHONY: clean cleanest

clean:
	rm -f *.o *.so *.pcm *.rootmap \
	rm -f DDASRootFit.cpp \
	converterargs.c converterargs.h \
	eeconverter
