# Compiler and flags:

CC = gcc
CXX = g++

FMTINC?=/usr/opt/ddastoys/DDASFormat/include
FMTLIB?=/usr/opt/ddastoys/DDASFormat/lib

# Assume DAQ environment is set up -- need DDAS verson with lddasfitformat!

CXXFLAGS = -Wall -g -O2 -fPIC -I.. -I$(DAQINC) -I$(FMTINC)
LDFLAGS = -L$(DAQLIB) -Wl,-rpath=$(DAQLIB) -L$(FMTLIB) -Wl,-rpath=$(FMTLIB)
LDLIBS = -ldataformat -lException -lurl -ldaqio -lFragmentIndex -lDDASFormat

# Flags related to CERN ROOT:

ROOTCFLAGS  = $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLDFLAGS = $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

all: eeconverter libDDASRootFit.so

eeconverter: eeconverter.o converterargs.o RootFileDataSink.o \
	ProcessToRootSink.o CEventProcessor.o StringsToIntegers.o \
	DDASRootFitHit.o DDASRootFitEvent.o DDASRootFit.o \
	../DDASFitHitUnpacker.o ../CRingItemProcessor.o
	$(CXX) $(CXXFLAGS) $(ROOTCFLAGS) -o $@ $^ \
	$(LDFLAGS) $(LDLIBS) $(ROOTLDFLAGS)

libDDASRootFit.so: DDASRootFit.o DDASRootFitHit.o DDASRootFitEvent.o
	$(CXX) -fPIC -shared -o $@ $(CXXFLAGS) $^ -L$(FMTLIB) \
	-Wl,-rpath=$(FMTLIB) -lDDASFormat

DDASRootFit.cpp: DDASRootFitEvent.h DDASRootFitHit.h ../DDASFitHit.h \
	RootExtensions.h LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ \
	-rml libDDASRootFit.so -rmf libDDASRootFit.rootmap \
	-I.. -I$(FMTINC) $^

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
