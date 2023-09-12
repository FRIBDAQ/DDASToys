# Compiler and flags:

CC = gcc
CXX = g++

# Assume DAQ environment is set up -- need DDAS verson with lddasfitformat
# and the unified format library!

CXXFLAGS = -Wall -g -O2 -fPIC -I.. -I$(DAQINC)
LDFLAGS = -L$(DAQLIB) -Wl,-rpath=$(DAQLIB)
LDLIBS = -ldataformat -lddasformat -lurl -lFragmentIndex -ldaqio

UFMT_INC = $(DAQROOT)/unifiedformat/include
UFMT_LIB = $(DAQROOT)/unifiedformat/lib
UFMT_CXXFLAGS = -I$(UFMT_INC)
UFMT_LDFLAGS = -L$(UFMT_LIB) -Wl,-rpath=$(UFMT_LIB)
UFMT_LDLIBS = -lNSCLDAQFormat -lV10Format -lV11Format -lV12Format \
	-lAbstractFormat

# Flags related to CERN ROOT:

ROOTCFLAGS  = $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLDFLAGS = $(shell $(ROOTSYS)/bin/root-config --libs --ldflags)

all: eeconverter libDDASRootFit.so

eeconverter: eeconverter.o converterargs.o CEventProcessor.o DataSource.o StreamDataSource.o ../CRingItemProcessor.o RootSinkProcessor.o RootFileDataSink.o DDASRootFitEvent.o DDASRootFitHit.o DDASRootFit.o RootExtensions.o
	$(CXX) $(UFMT_CXXFLAGS) $(CXXFLAGS) $(ROOTCFLAGS) -o $@ $^ \
	$(UFMT_LDFLAGS) $(UFMT_LDLIBS) $(LDFLAGS) $(LDLIBS) $(ROOTLDFLAGS)

libDDASRootFit.so: DDASRootFit.o DDASRootFitHit.o DDASRootFitEvent.o \
	RootExtensions.o
	$(CXX) -fPIC -shared -o $@ $(CXXFLAGS) $^ $(LDFLAGS) -lddasformat

DDASRootFit.cpp: DDASRootFitEvent.h DDASRootFitHit.h ../DDASFitHit.h \
	RootExtensions.h LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ \
	-rml libDDASRootFit.so -rmf libDDASRootFit.rootmap \
	-I$(DAQINC) $^

%.o : %.cpp
	$(CXX) -c $(UFMT_CXXFLAGS) $(CXXFLAGS) $(ROOTCFLAGS) $<

converterargs.o:
	gengetopt < converterargs.ggo --file-name=converterargs
	$(CC) -c converterargs.c

.PHONY: clean cleanest

clean:
	rm -f *.o *.so *.pcm *.rootmap \
	rm -f DDASRootFit.cpp \
	converterargs.c converterargs.h \
	eeconverter
